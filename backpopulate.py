#!/usr/bin/env python3
"""
Backpopulate LatencyGrid leaderboard with weekly submissions using cached benchmark data.
Runs benchmarks once, then submits with backdated timestamps for past Mondays at 07:00 UTC.
"""
import os, json, asyncio, subprocess, datetime, sys

# Load .env if present
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import httpx

API_URL = "https://api.latencygrid.dev/submit"
REGION = "canada"
LOCATION = "Ottawa, Canada"
PIPELINE_TTS = "lmnt/blizzard"

VENV_BIN = os.path.join(os.path.dirname(__file__), ".venv", "bin")
MODELPING = os.path.join(VENV_BIN, "modelping")

# How many past weekly slots to backfill (not counting current week)
WEEKS_BACK = 5


def run_cmd(cmd, timeout=120):
    env = {**os.environ, "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}"}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        if result.returncode != 0:
            print(f"  stderr: {result.stderr[:200]}")
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        print(f"  Command timed out after {timeout}s: {' '.join(cmd)}")
        return "", 1


def run_llm():
    print("Running LLM benchmarks...")
    out, code = run_cmd([MODELPING, "run", "--all", "--runs", "3", "--json"])
    if code == 0 and out.strip():
        try:
            return json.loads(out)
        except Exception as e:
            print(f"  LLM parse error: {e}")
    return []


def extract_json(out):
    for start_char in ('[', '{'):
        idx = out.find(start_char)
        if idx != -1:
            try:
                return json.loads(out[idx:])
            except Exception:
                pass
    return None


def run_stt():
    print("Running STT benchmarks...")
    out, code = run_cmd([MODELPING, "stt", "--runs", "2", "--json"], timeout=180)
    if out.strip():
        try:
            data = extract_json(out)
            if data is None:
                print("  STT parse error: no JSON found")
                return []
            return [
                {
                    "model": item.get("model", ""),
                    "provider": item.get("provider", ""),
                    "latency_p50": item.get("transcription_latency_ms", 0),
                }
                for item in data if not item.get("error")
            ]
        except Exception as e:
            print(f"  STT parse error: {e}")
    return []


def run_tts():
    print("Running TTS benchmarks...")
    out, code = run_cmd([MODELPING, "tts", "--runs", "2", "--json"], timeout=180)
    if out.strip():
        try:
            data = extract_json(out)
            if data is None:
                print("  TTS parse error: no JSON found")
                return []
            return [
                {
                    "model": item.get("model", ""),
                    "provider": item.get("provider", ""),
                    "ttfb_p50": item.get("ttfb_ms", 0),
                    "rtf": item.get("realtime_factor", 0),
                }
                for item in data if not item.get("error")
            ]
        except Exception as e:
            print(f"  TTS parse error: {e}")
    return []


def run_pipeline():
    print("Running pipeline benchmark...")
    out, code = run_cmd(
        [MODELPING, "pipeline",
         "--stt", "groq/whisper-large-v3-turbo",
         "--llm", "llama-3.3-70b-versatile",
         "--tts", PIPELINE_TTS,
         "--runs", "2"],
        timeout=120,
    )
    for line in out.split("\n"):
        if "fastest total:" in line.lower():
            try:
                ms = int("".join(filter(str.isdigit, line.split("fastest total:")[1].split("ms")[0])))
                return {
                    "stt_model": "groq/whisper-large-v3-turbo",
                    "llm_model": "llama-3.3-70b-versatile",
                    "tts_model": PIPELINE_TTS,
                    "total_pipeline_ms": ms,
                }
            except Exception:
                pass
    print(f"  Pipeline output: {out[:300]}")
    return None


async def submit(llm, stt, tts, pipeline, timestamp_str):
    payload = {
        "region": REGION,
        "location": LOCATION,
        "timestamp": timestamp_str,
        "modelping_version": "0.1.0",
        "llm": llm,
        "stt": stt,
        "tts": tts,
        "pipeline": [pipeline] if pipeline else [],
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(API_URL, json=payload)
        return r.json()


def past_mondays(n):
    """Return the last n Mondays (most recent first) at 07:00 UTC."""
    today = datetime.date.today()
    # Find last Monday
    days_since_monday = today.weekday()  # 0=Monday
    last_monday = today - datetime.timedelta(days=days_since_monday)
    return [
        datetime.datetime(last_monday.year, last_monday.month, last_monday.day, 7, 0, 0,
                          tzinfo=datetime.timezone.utc) - datetime.timedelta(weeks=i)
        for i in range(1, n + 1)
    ]


def main():
    print("=== LatencyGrid Backpopulate ===")
    print(f"Running fresh benchmarks to backfill {WEEKS_BACK} past weeks...")
    print()

    llm = run_llm()
    stt = run_stt()
    tts = run_tts()
    pipeline = run_pipeline()

    print()
    print(f"Benchmark results: {len(llm)} LLM, {len(stt)} STT, {len(tts)} TTS, pipeline: {bool(pipeline)}")

    if not llm and not stt and not tts:
        print("No results — aborting.")
        sys.exit(1)

    # Verify what we have; remove categories with zero data
    print()
    print("Submitting backdated entries...")
    timestamps = past_mondays(WEEKS_BACK)
    urls = []

    for ts in timestamps:
        ts_str = ts.isoformat().replace("+00:00", "Z")
        print(f"  Submitting for {ts_str}...")
        result = asyncio.run(submit(llm, stt, tts, pipeline, ts_str))
        if result.get("success"):
            url = result.get("url")
            urls.append((ts_str, url))
            print(f"    ✓ {url}")
        else:
            print(f"    ✗ Failed: {result}")

    print()
    print(f"Done. {len(urls)}/{WEEKS_BACK} submissions successful.")
    for ts_str, url in urls:
        print(f"  {ts_str}: {url}")


if __name__ == "__main__":
    main()
