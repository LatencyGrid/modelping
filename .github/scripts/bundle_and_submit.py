"""Bundle benchmark results and submit to LatencyGrid leaderboard."""
import json, datetime, asyncio
import httpx

with open('llm-results.json') as f:
    llm = json.load(f)
with open('stt-results.json') as f:
    stt = json.load(f)
with open('tts-results.json') as f:
    tts = json.load(f)

bundle = {
    'region': 'us-east',
    'location': 'US East (Azure)',
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'modelping_version': '0.1.0',
    'llm': llm,
    'stt': stt,
    'tts': tts,
}

with open('benchmark-results.json', 'w') as f:
    json.dump(bundle, f, indent=2)

print(f"Bundled: {len(llm)} LLM, {len(stt)} STT, {len(tts)} TTS results")

async def submit():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post('https://api.latencygrid.dev/submit', json=bundle)
        return r.json()

result = asyncio.run(submit())
print('Submitted:', result.get('success'), result.get('url', result.get('error', '')))
