"""Audio utility functions."""

from __future__ import annotations

import wave
from pathlib import Path


def get_audio_duration_ms(audio_path: str) -> float:
    """Return duration of a WAV file in milliseconds."""
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return (frames / rate) * 1000.0
    except Exception:
        return 0.0


def get_test_audio_path() -> str:
    """Return path to the bundled test audio file."""
    return str(Path(__file__).parent.parent / "test_audio" / "test_speech.wav")
