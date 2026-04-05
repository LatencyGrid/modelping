"""Tests for STT and TTS providers, metrics, and pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
import httpx

from modelping.models import STTRunResult, TTSRunResult, PipelineRunResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _async_iter(items):
    """Return an async iterator over bytes items."""
    async def _gen():
        for item in items:
            yield item
    return _gen()


class MockStreamResponse:
    """Mock for httpx streaming context manager."""

    def __init__(self, chunks: list[bytes] = None, lines: list[str] = None, status_code: int = 200):
        self._chunks = chunks or []
        self._lines = lines or []
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            request = MagicMock()
            response = MagicMock()
            response.status_code = self.status_code
            response.text = "Error"
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def aiter_bytes(self, chunk_size=None):
        return _async_iter(self._chunks)

    def aiter_lines(self):
        return _async_iter(self._lines)


class MockSyncResponse:
    """Mock for non-streaming httpx response."""

    def __init__(self, json_data: dict = None, status_code: int = 200):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = json.dumps(json_data or {})

    def raise_for_status(self):
        if self.status_code >= 400:
            request = MagicMock()
            response = MagicMock()
            response.status_code = self.status_code
            response.text = "Error"
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._json_data


class MockAsyncClient:
    """Mock httpx.AsyncClient that supports both stream() and regular post()."""

    def __init__(
        self,
        stream_response: MockStreamResponse = None,
        post_responses: list[MockSyncResponse] = None,
        get_responses: list[MockSyncResponse] = None,
    ):
        self._stream = stream_response
        self._post_responses = iter(post_responses or [])
        self._get_responses = iter(get_responses or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, *args, **kwargs):
        return self._stream

    async def post(self, *args, **kwargs):
        return next(self._post_responses)

    async def get(self, *args, **kwargs):
        return next(self._get_responses)


# ── STT Models ────────────────────────────────────────────────────────────────

class TestSTTRunResult:
    def test_basic_construction(self):
        r = STTRunResult(
            provider="groq_stt",
            model="whisper-large-v3",
            audio_duration_ms=5000.0,
            transcription_latency_ms=320.0,
            word_count=9,
        )
        assert r.provider == "groq_stt"
        assert r.model == "whisper-large-v3"
        assert r.audio_duration_ms == 5000.0
        assert r.transcription_latency_ms == 320.0
        assert r.word_count == 9
        assert r.error is None
        assert r.ttft_ms is None

    def test_error_result(self):
        r = STTRunResult(
            provider="groq_stt",
            model="test",
            audio_duration_ms=0.0,
            transcription_latency_ms=0.0,
            word_count=0,
            error="API key missing",
        )
        assert r.error == "API key missing"

    def test_with_ttft(self):
        r = STTRunResult(
            provider="openai_stt",
            model="whisper-1",
            audio_duration_ms=5000.0,
            transcription_latency_ms=400.0,
            ttft_ms=150.0,
            word_count=9,
        )
        assert r.ttft_ms == 150.0


# ── TTS Models ────────────────────────────────────────────────────────────────

class TestTTSRunResult:
    def test_basic_construction(self):
        r = TTSRunResult(
            provider="cartesia_tts",
            model="sonic-2",
            text_chars=90,
            ttfb_ms=89.0,
            total_ms=800.0,
            audio_duration_ms=3200.0,
            realtime_factor=4.0,
        )
        assert r.provider == "cartesia_tts"
        assert r.ttfb_ms == 89.0
        assert r.realtime_factor == 4.0
        assert r.error is None

    def test_realtime_factor(self):
        r = TTSRunResult(
            provider="test",
            model="test",
            text_chars=50,
            ttfb_ms=50.0,
            total_ms=1000.0,
            audio_duration_ms=5000.0,
            realtime_factor=5.0,
        )
        assert r.realtime_factor > 1.0  # faster than real-time


# ── Pipeline Models ───────────────────────────────────────────────────────────

class TestPipelineRunResult:
    def test_basic_construction(self):
        r = PipelineRunResult(
            stt_provider="groq_stt",
            stt_model="groq/whisper-large-v3",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            tts_provider="cartesia_tts",
            tts_model="cartesia/sonic-2",
            stt_latency_ms=180.0,
            llm_ttft_ms=42.0,
            tts_ttfb_ms=89.0,
            total_pipeline_ms=311.0,
        )
        assert r.total_pipeline_ms == 311.0
        assert r.error is None

    def test_error_result(self):
        r = PipelineRunResult(
            stt_provider="groq_stt",
            stt_model="groq/whisper-large-v3",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            tts_provider="cartesia_tts",
            tts_model="cartesia/sonic-2",
            stt_latency_ms=0.0,
            llm_ttft_ms=0.0,
            tts_ttfb_ms=0.0,
            total_pipeline_ms=0.0,
            error="STT API key missing",
        )
        assert r.error is not None


# ── BaseSTTProvider ───────────────────────────────────────────────────────────

class TestBaseSTTProvider:
    def test_make_result(self):
        from modelping.providers.stt.groq_stt import GroqSTTProvider
        provider = GroqSTTProvider()
        r = provider._make_result(
            model="whisper-large-v3",
            audio_duration_ms=5000.0,
            transcription_latency_ms=300.0,
            word_count=9,
        )
        assert r.provider == "groq_stt"
        assert r.transcription_latency_ms == 300.0
        assert r.error is None

    def test_error_result(self):
        from modelping.providers.stt.groq_stt import GroqSTTProvider
        provider = GroqSTTProvider()
        r = provider._error_result("whisper-large-v3", 5000.0, "network error")
        assert r.error == "network error"
        assert r.transcription_latency_ms == 0.0
        assert r.word_count == 0


# ── BaseTTSProvider ───────────────────────────────────────────────────────────

class TestBaseTTSProvider:
    def test_make_result_realtime_factor(self):
        from modelping.providers.tts.cartesia_tts import CartesiaTTSProvider
        provider = CartesiaTTSProvider()
        r = provider._make_result(
            model="sonic-2",
            text_chars=90,
            ttfb_ms=89.0,
            total_ms=500.0,
            audio_duration_ms=2500.0,
        )
        assert r.realtime_factor == pytest.approx(5.0)
        assert r.provider == "cartesia_tts"

    def test_error_result(self):
        from modelping.providers.tts.openai_tts import OpenAITTSProvider
        provider = OpenAITTSProvider()
        r = provider._error_result("tts-1", 90, "connection refused")
        assert r.error == "connection refused"
        assert r.ttfb_ms == 0.0

    def test_realtime_factor_zero_total(self):
        from modelping.providers.tts.elevenlabs_tts import ElevenLabsTTSProvider
        provider = ElevenLabsTTSProvider()
        r = provider._make_result(
            model="eleven_flash_v2_5",
            text_chars=50,
            ttfb_ms=0.0,
            total_ms=0.0,
            audio_duration_ms=1000.0,
        )
        assert r.realtime_factor == 0.0  # avoid division by zero


# ── Groq STT Provider ─────────────────────────────────────────────────────────

class TestGroqSTTProvider:
    @pytest.mark.asyncio
    async def test_successful_transcribe(self, monkeypatch, tmp_path):
        from modelping.providers.stt.groq_stt import GroqSTTProvider

        # Create a minimal WAV file
        import wave, struct
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

        mock_resp = MockSyncResponse({"text": "The quick brown fox jumps"})
        mock_client = MockAsyncClient(post_responses=[mock_resp])

        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        with patch("modelping.providers.stt.groq_stt.httpx.AsyncClient", return_value=mock_client):
            provider = GroqSTTProvider()
            result = await provider.transcribe(wav_path, "whisper-large-v3")

        assert result.error is None
        assert result.provider == "groq_stt"
        assert result.word_count == 5
        assert result.transcription_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.stt.groq_stt import GroqSTTProvider
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqSTTProvider()
        result = await provider.transcribe("/fake/path.wav", "whisper-large-v3")
        assert result.error is not None
        assert "GROQ_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch, tmp_path):
        from modelping.providers.stt.groq_stt import GroqSTTProvider
        import wave, struct
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 1000, *([0] * 1000)))

        mock_resp = MockSyncResponse({}, status_code=401)
        mock_client = MockAsyncClient(post_responses=[mock_resp])

        monkeypatch.setenv("GROQ_API_KEY", "bad-key")
        with patch("modelping.providers.stt.groq_stt.httpx.AsyncClient", return_value=mock_client):
            provider = GroqSTTProvider()
            result = await provider.transcribe(wav_path, "whisper-large-v3")

        assert result.error is not None
        assert "401" in result.error


# ── OpenAI STT Provider ───────────────────────────────────────────────────────

class TestOpenAISTTProvider:
    @pytest.mark.asyncio
    async def test_successful_transcribe(self, monkeypatch, tmp_path):
        from modelping.providers.stt.openai_stt import OpenAISTTProvider
        import wave, struct
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

        mock_resp = MockSyncResponse({"text": "Hello world from OpenAI"})
        mock_client = MockAsyncClient(post_responses=[mock_resp])

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("modelping.providers.stt.openai_stt.httpx.AsyncClient", return_value=mock_client):
            provider = OpenAISTTProvider()
            result = await provider.transcribe(wav_path, "whisper-1")

        assert result.error is None
        assert result.provider == "openai_stt"
        assert result.word_count == 4

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.stt.openai_stt import OpenAISTTProvider
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAISTTProvider()
        result = await provider.transcribe("/fake.wav", "whisper-1")
        assert "OPENAI_API_KEY" in result.error


# ── Deepgram STT Provider ─────────────────────────────────────────────────────

class TestDeepgramSTTProvider:
    @pytest.mark.asyncio
    async def test_successful_transcribe(self, monkeypatch, tmp_path):
        from modelping.providers.stt.deepgram_stt import DeepgramSTTProvider
        import wave, struct
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

        deepgram_response = {
            "results": {
                "channels": [{
                    "alternatives": [{"transcript": "The quick brown fox"}]
                }]
            }
        }
        mock_resp = MockSyncResponse(deepgram_response)
        mock_client = MockAsyncClient(post_responses=[mock_resp])

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        with patch("modelping.providers.stt.deepgram_stt.httpx.AsyncClient", return_value=mock_client):
            provider = DeepgramSTTProvider()
            result = await provider.transcribe(wav_path, "nova-2")

        assert result.error is None
        assert result.word_count == 4

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.stt.deepgram_stt import DeepgramSTTProvider
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        provider = DeepgramSTTProvider()
        result = await provider.transcribe("/fake.wav", "nova-2")
        assert "DEEPGRAM_API_KEY" in result.error


# ── AssemblyAI STT Provider ───────────────────────────────────────────────────

class TestAssemblyAISTTProvider:
    @pytest.mark.asyncio
    async def test_successful_transcribe(self, monkeypatch, tmp_path):
        from modelping.providers.stt.assemblyai_stt import AssemblyAISTTProvider
        import wave, struct
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

        upload_resp = MockSyncResponse({"upload_url": "https://cdn.assemblyai.com/upload/test"})
        submit_resp = MockSyncResponse({"id": "abc123"})
        poll_resp = MockSyncResponse({"status": "completed", "text": "The quick brown fox jumps over"})

        mock_client = MockAsyncClient(
            post_responses=[upload_resp, submit_resp],
            get_responses=[poll_resp],
        )

        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
        with patch("modelping.providers.stt.assemblyai_stt.httpx.AsyncClient", return_value=mock_client):
            with patch("modelping.providers.stt.assemblyai_stt.asyncio.sleep", return_value=None):
                provider = AssemblyAISTTProvider()
                result = await provider.transcribe(wav_path, "best")

        assert result.error is None
        assert result.word_count == 6

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.stt.assemblyai_stt import AssemblyAISTTProvider
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        provider = AssemblyAISTTProvider()
        result = await provider.transcribe("/fake.wav", "best")
        assert "ASSEMBLYAI_API_KEY" in result.error


# ── Gladia STT Provider ───────────────────────────────────────────────────────

class TestGladiaSTTProvider:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.stt.gladia_stt import GladiaSTTProvider
        monkeypatch.delenv("GLADIA_API_KEY", raising=False)
        provider = GladiaSTTProvider()
        result = await provider.transcribe("/fake.wav", "default")
        assert "GLADIA_API_KEY" in result.error


# ── ElevenLabs TTS Provider ───────────────────────────────────────────────────

class TestElevenLabsTTSProvider:
    @pytest.mark.asyncio
    async def test_successful_synthesize(self, monkeypatch):
        from modelping.providers.tts.elevenlabs_tts import ElevenLabsTTSProvider

        audio_chunks = [b"\x00\x01" * 500, b"\x02\x03" * 500]
        mock_resp = MockStreamResponse(chunks=audio_chunks)
        mock_client = MockAsyncClient(stream_response=mock_resp)

        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        with patch("modelping.providers.tts.elevenlabs_tts.httpx.AsyncClient", return_value=mock_client):
            provider = ElevenLabsTTSProvider()
            result = await provider.synthesize("Hello world", "eleven_flash_v2_5")

        assert result.error is None
        assert result.provider == "elevenlabs_tts"
        assert result.ttfb_ms >= 0
        assert result.total_ms >= 0
        assert result.text_chars == 11

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.elevenlabs_tts import ElevenLabsTTSProvider
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        provider = ElevenLabsTTSProvider()
        result = await provider.synthesize("Hello", "eleven_flash_v2_5")
        assert "ELEVENLABS_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch):
        from modelping.providers.tts.elevenlabs_tts import ElevenLabsTTSProvider
        mock_resp = MockStreamResponse(chunks=[], status_code=401)
        mock_client = MockAsyncClient(stream_response=mock_resp)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "bad-key")
        with patch("modelping.providers.tts.elevenlabs_tts.httpx.AsyncClient", return_value=mock_client):
            provider = ElevenLabsTTSProvider()
            result = await provider.synthesize("Hello", "eleven_flash_v2_5")
        assert result.error is not None
        assert "401" in result.error


# ── Cartesia TTS Provider ─────────────────────────────────────────────────────

class TestCartesiaTTSProvider:
    @pytest.mark.asyncio
    async def test_successful_synthesize(self, monkeypatch):
        from modelping.providers.tts.cartesia_tts import CartesiaTTSProvider

        # PCM f32le data (4 bytes per sample at 44100 Hz = ~1 sec)
        audio_chunks = [b"\x00" * 44100 * 4]
        mock_resp = MockStreamResponse(chunks=audio_chunks)
        mock_client = MockAsyncClient(stream_response=mock_resp)

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        with patch("modelping.providers.tts.cartesia_tts.httpx.AsyncClient", return_value=mock_client):
            provider = CartesiaTTSProvider()
            result = await provider.synthesize("Hello world", "sonic-2")

        assert result.error is None
        assert result.provider == "cartesia_tts"
        assert result.audio_duration_ms == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.cartesia_tts import CartesiaTTSProvider
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        provider = CartesiaTTSProvider()
        result = await provider.synthesize("Hello", "sonic-2")
        assert "CARTESIA_API_KEY" in result.error


# ── OpenAI TTS Provider ───────────────────────────────────────────────────────

class TestOpenAITTSProvider:
    @pytest.mark.asyncio
    async def test_successful_synthesize(self, monkeypatch):
        from modelping.providers.tts.openai_tts import OpenAITTSProvider

        audio_chunks = [b"\xff\xfb" * 1000]
        mock_resp = MockStreamResponse(chunks=audio_chunks)
        mock_client = MockAsyncClient(stream_response=mock_resp)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("modelping.providers.tts.openai_tts.httpx.AsyncClient", return_value=mock_client):
            provider = OpenAITTSProvider()
            result = await provider.synthesize("Test text", "tts-1")

        assert result.error is None
        assert result.provider == "openai_tts"
        assert result.ttfb_ms >= 0

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.openai_tts import OpenAITTSProvider
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAITTSProvider()
        result = await provider.synthesize("Hello", "tts-1")
        assert "OPENAI_API_KEY" in result.error


# ── Deepgram TTS Provider ─────────────────────────────────────────────────────

class TestDeepgramTTSProvider:
    @pytest.mark.asyncio
    async def test_successful_synthesize(self, monkeypatch):
        from modelping.providers.tts.deepgram_tts import DeepgramTTSProvider

        # 16-bit PCM at 24kHz for 1 second = 24000 * 2 bytes
        audio_chunks = [b"\x00" * 48000]
        mock_resp = MockStreamResponse(chunks=audio_chunks)
        mock_client = MockAsyncClient(stream_response=mock_resp)

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        with patch("modelping.providers.tts.deepgram_tts.httpx.AsyncClient", return_value=mock_client):
            provider = DeepgramTTSProvider()
            result = await provider.synthesize("Hello world", "aura-asteria-en")

        assert result.error is None
        assert result.audio_duration_ms == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.deepgram_tts import DeepgramTTSProvider
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        provider = DeepgramTTSProvider()
        result = await provider.synthesize("Hello", "aura-asteria-en")
        assert "DEEPGRAM_API_KEY" in result.error


# ── LMNT TTS Provider ─────────────────────────────────────────────────────────

class TestLMNTTTSProvider:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.lmnt_tts import LMNTTTSProvider
        monkeypatch.delenv("LMNT_API_KEY", raising=False)
        provider = LMNTTTSProvider()
        result = await provider.synthesize("Hello", "blizzard")
        assert "LMNT_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_successful_synthesize(self, monkeypatch):
        from modelping.providers.tts.lmnt_tts import LMNTTTSProvider

        audio_chunks = [b"\xff\xfb" * 800]
        mock_resp = MockStreamResponse(chunks=audio_chunks)
        mock_client = MockAsyncClient(stream_response=mock_resp)

        monkeypatch.setenv("LMNT_API_KEY", "test-key")
        with patch("modelping.providers.tts.lmnt_tts.httpx.AsyncClient", return_value=mock_client):
            provider = LMNTTTSProvider()
            result = await provider.synthesize("Hello world", "blizzard")

        assert result.error is None
        assert result.provider == "lmnt_tts"


# ── PlayHT TTS Provider ───────────────────────────────────────────────────────

class TestPlayHTTTSProvider:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.playht_tts import PlayHTTTSProvider
        monkeypatch.delenv("PLAYHT_API_KEY", raising=False)
        monkeypatch.delenv("PLAYHT_USER_ID", raising=False)
        provider = PlayHTTTSProvider()
        result = await provider.synthesize("Hello", "PlayDialog")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_user_id(self, monkeypatch):
        from modelping.providers.tts.playht_tts import PlayHTTTSProvider
        monkeypatch.setenv("PLAYHT_API_KEY", "test-key")
        monkeypatch.delenv("PLAYHT_USER_ID", raising=False)
        provider = PlayHTTTSProvider()
        result = await provider.synthesize("Hello", "PlayDialog")
        assert result.error is not None
        assert "PLAYHT_USER_ID" in result.error


# ── Fish Audio TTS Provider ───────────────────────────────────────────────────

class TestFishAudioTTSProvider:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.tts.fish_audio_tts import FishAudioTTSProvider
        monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
        provider = FishAudioTTSProvider()
        result = await provider.synthesize("Hello", "default")
        assert "FISH_AUDIO_API_KEY" in result.error


# ── Audio Utilities ───────────────────────────────────────────────────────────

class TestAudioUtils:
    def test_get_audio_duration_ms(self, tmp_path):
        import wave, struct
        from modelping.utils.audio import get_audio_duration_ms

        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

        duration = get_audio_duration_ms(wav_path)
        assert duration == pytest.approx(1000.0)

    def test_invalid_file(self):
        from modelping.utils.audio import get_audio_duration_ms
        duration = get_audio_duration_ms("/nonexistent/file.wav")
        assert duration == 0.0

    def test_get_test_audio_path_exists(self):
        import os
        from modelping.utils.audio import get_test_audio_path
        path = get_test_audio_path()
        assert path.endswith("test_speech.wav")
        assert os.path.exists(path)


# ── Pipeline Orchestration ────────────────────────────────────────────────────

class TestPipelineOrchestration:
    @pytest.mark.asyncio
    async def test_pipeline_with_missing_stt_key(self, monkeypatch):
        from modelping.pipeline_runner import run_pipeline
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        result = await run_pipeline(
            "groq/whisper-large-v3",
            "gpt-4o-mini",
            "cartesia/sonic-2",
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_missing_llm_key(self, monkeypatch):
        from modelping.pipeline_runner import run_pipeline
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = await run_pipeline(
            "groq/whisper-large-v3",
            "gpt-4o-mini",
            "cartesia/sonic-2",
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_missing_tts_key(self, monkeypatch):
        from modelping.pipeline_runner import run_pipeline
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        result = await run_pipeline(
            "groq/whisper-large-v3",
            "gpt-4o-mini",
            "cartesia/sonic-2",
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_pipeline_unknown_stt_model(self, monkeypatch):
        from modelping.pipeline_runner import run_pipeline
        result = await run_pipeline("bad/model", "gpt-4o-mini", "cartesia/sonic-2")
        assert result.error is not None
        assert "Unknown STT" in result.error

    @pytest.mark.asyncio
    async def test_run_pipeline_matrix(self, monkeypatch):
        from modelping.pipeline_runner import run_pipeline_matrix

        # All keys missing — should return errors without crashing
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)

        results = await run_pipeline_matrix(
            ["groq/whisper-large-v3"],
            ["gpt-4o-mini"],
            ["cartesia/sonic-2"],
            runs=1,
        )
        assert len(results) == 1
        assert results[0].error is not None


# ── STT Metrics Calculations ──────────────────────────────────────────────────

class TestSTTMetrics:
    def test_real_time_factor_calculation(self):
        """RTF = latency / audio_duration; <1 means faster than real-time."""
        latency_ms = 320.0
        audio_ms = 5000.0
        rtf = latency_ms / audio_ms
        assert rtf == pytest.approx(0.064)

    def test_word_count_from_transcript(self):
        text = "The quick brown fox jumps over the lazy dog."
        word_count = len(text.split())
        assert word_count == 9

    def test_empty_transcript_word_count(self):
        text = ""
        word_count = len(text.split()) if text.strip() else 0
        assert word_count == 0


# ── TTS Metrics Calculations ──────────────────────────────────────────────────

class TestTTSMetrics:
    def test_realtime_factor_above_one(self):
        """Faster-than-real-time: audio_duration > total_ms."""
        audio_duration_ms = 5000.0
        total_ms = 800.0
        rtf = audio_duration_ms / total_ms
        assert rtf > 1.0
        assert rtf == pytest.approx(6.25)

    def test_audio_duration_from_pcm(self):
        """PCM f32le at 44100 Hz: 4 bytes/sample."""
        total_bytes = 44100 * 4  # 1 second
        num_samples = total_bytes / 4
        duration_ms = (num_samples / 44100) * 1000
        assert duration_ms == pytest.approx(1000.0)

    def test_audio_duration_from_mp3_estimate(self):
        """MP3 at 128kbps: bytes * 8 / 128000 * 1000."""
        total_bytes = 128_000 // 8  # 1 second of 128kbps MP3
        duration_ms = (total_bytes * 8 / 128_000) * 1000
        assert duration_ms == pytest.approx(1000.0)

    def test_ttfb_is_less_than_total(self):
        """TTFB should always be ≤ total time."""
        r = TTSRunResult(
            provider="test",
            model="test",
            text_chars=50,
            ttfb_ms=89.0,
            total_ms=500.0,
            audio_duration_ms=2000.0,
            realtime_factor=4.0,
        )
        assert r.ttfb_ms <= r.total_ms
