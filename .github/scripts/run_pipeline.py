"""Run pipeline benchmark and write results to pipeline-results.json."""
import asyncio, json, sys, os
sys.stderr = open(os.devnull, 'w')

from modelping.pipeline_runner import run_pipeline
from modelping.config import get_api_key, get_stt_api_key, get_tts_api_key

async def run():
    if not (get_api_key('groq') and get_tts_api_key('cartesia_tts')):
        print("Skipping pipeline — missing groq or cartesia keys")
        return []
    
    results = []
    for _ in range(3):
        try:
            r = await run_pipeline(
                stt_model_key="groq/whisper-large-v3-turbo",
                llm_model_key="llama-3.3-70b-versatile",
                tts_model_key="cartesia/sonic-2",
            )
            if not r.error:
                results.append(r)
        except Exception as e:
            print(f"Pipeline run error: {e}")
    
    if not results:
        return []
    
    # P50 of total_pipeline_ms
    times = sorted(r.total_pipeline_ms for r in results)
    best = results[0]
    return [{
        "stt_model": best.stt_model,
        "stt_provider": best.stt_provider,
        "llm_model": best.llm_model,
        "llm_provider": best.llm_provider,
        "tts_model": best.tts_model,
        "tts_provider": best.tts_provider,
        "stt_latency_ms": round(sorted(r.stt_latency_ms for r in results)[len(results)//2]),
        "llm_ttft_ms": round(sorted(r.llm_ttft_ms for r in results)[len(results)//2]),
        "tts_ttfb_ms": round(sorted(r.tts_ttfb_ms for r in results)[len(results)//2]),
        "total_pipeline_ms": round(times[len(times)//2]),
        "runs": len(results)
    }]

with open('pipeline-results.json', 'w') as f:
    json.dump(asyncio.run(run()), f)

data = json.load(open('pipeline-results.json'))
print(f"Pipeline: {len(data)} results" + (f" — fastest: {data[0]['total_pipeline_ms']}ms" if data else ""))
