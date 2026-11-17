import os
from pathlib import Path

from psyconassist.engine import whisper as wh

print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
print("WHISPER_CACHE", os.environ.get("WHISPER_CACHE"))
wav = Path("data/S20261117-001/audio/16k.wav")
segs = wh.transcribe(wav, model_size="small", language="zh")
print("n_segs", len(segs))
for s in segs[:12]:
    print(f"{s['t0']:.1f}-{s['t1']:.1f}: {s['text']}")
