"""Run TTS benchmarks and write results to tts-results.json."""
import asyncio, json, sys, os

sys.stderr = open(os.devnull, 'w')

from modelping.config import TTS_MODELS, get_tts_api_key, TTS_DEFAULT_TEXT
from modelping.providers.tts.elevenlabs_tts import ElevenLabsTTSProvider
from modelping.providers.tts.cartesia_tts import CartesiaTTSProvider
from modelping.providers.tts.openai_tts import OpenAITTSProvider
from modelping.providers.tts.deepgram_tts import DeepgramTTSProvider
from modelping.providers.tts.fish_audio_tts import FishAudioTTSProvider
from modelping.providers.tts.lmnt_tts import LMNTTTSProvider

PROVIDER_MAP = {
    'elevenlabs_tts': ElevenLabsTTSProvider,
    'cartesia_tts': CartesiaTTSProvider,
    'openai_tts': OpenAITTSProvider,
    'deepgram_tts': DeepgramTTSProvider,
    'fish_audio_tts': FishAudioTTSProvider,
    'lmnt_tts': LMNTTTSProvider,
}

async def run():
    results = []
    for key, cfg in TTS_MODELS.items():
        p_name = cfg['provider']
        if not get_tts_api_key(p_name):
            continue
        cls = PROVIDER_MAP.get(p_name)
        if not cls:
            continue
        runs = []
        for _ in range(3):
            r = await cls().synthesize(TTS_DEFAULT_TEXT, cfg['model_id'])
            if not r.error:
                runs.append(r.ttfb_ms)
        if runs:
            results.append({'model': key, 'provider': p_name, 'ttfb_p50': sorted(runs)[len(runs)//2]})
    return results

with open('tts-results.json', 'w') as f:
    json.dump(asyncio.run(run()), f)

print(f"TTS: {len(json.load(open('tts-results.json')))} models")
