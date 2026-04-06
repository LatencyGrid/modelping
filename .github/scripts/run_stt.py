"""Run STT benchmarks and write results to stt-results.json."""
import asyncio, json, sys, os

sys.stderr = open(os.devnull, 'w')

from modelping.config import STT_MODELS, get_stt_api_key
from modelping.providers.stt.groq_stt import GroqSTTProvider
from modelping.providers.stt.openai_stt import OpenAISTTProvider
from modelping.providers.stt.deepgram_stt import DeepgramSTTProvider
from modelping.providers.stt.assemblyai_stt import AssemblyAISTTProvider
from modelping.providers.stt.gladia_stt import GladiaSTTProvider
from modelping.utils.audio import get_test_audio_path

PROVIDER_MAP = {
    'groq_stt': GroqSTTProvider,
    'openai_stt': OpenAISTTProvider,
    'deepgram_stt': DeepgramSTTProvider,
    'assemblyai_stt': AssemblyAISTTProvider,
    'gladia_stt': GladiaSTTProvider,
}

async def run():
    audio = get_test_audio_path()
    results = []
    for key, cfg in STT_MODELS.items():
        p_name = cfg['provider']
        if not get_stt_api_key(p_name):
            continue
        cls = PROVIDER_MAP.get(p_name)
        if not cls:
            continue
        runs = []
        for _ in range(3):
            r = await cls().transcribe(audio, cfg['model_id'])
            if not r.error:
                runs.append(r.transcription_latency_ms)
        if runs:
            results.append({'model': key, 'provider': p_name, 'latency_p50': sorted(runs)[len(runs)//2]})
    return results

with open('stt-results.json', 'w') as f:
    json.dump(asyncio.run(run()), f)

print(f"STT: {len(json.load(open('stt-results.json')))} models")
