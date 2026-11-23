"""Zet een opname (m4a van de iPhone, mp3, mp4) om naar wav: 16 kHz, mono.

Gebruik:  .venv-asr\\Scripts\\python src\\m4a_naar_wav.py data\\opname_3min.m4a
"""
import sys
import wave

import numpy as np
from faster_whisper import decode_audio

bron = sys.argv[1]
doel = bron.rsplit(".", 1)[0] + ".wav"

audio = decode_audio(bron, sampling_rate=16000)  # 16 kHz, mono
pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

with wave.open(doel, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(pcm.tobytes())
print(f"Opgeslagen: {doel} ({len(pcm) / 16000:.1f} s)")
