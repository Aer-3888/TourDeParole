"""Transcribe AMI meetings with faster-whisper, keeping word timestamps.

    python -m tourdeparole.transcribe --split test
"""

import argparse
import json
import time
from pathlib import Path

import torch  # noqa: F401  loads the CUDA libraries that CTranslate2 needs on Windows
from faster_whisper import WhisperModel

from tourdeparole import ami

SETTINGS = {"language": "en", "beam_size": 5, "word_timestamps": True, "vad_filter": True}


def transcribe(model: WhisperModel, audio) -> list[dict]:
    segments, _ = model.transcribe(audio, **SETTINGS)
    return [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word.strip(), "probability": w.probability}
                for w in s.words
            ],
        }
        for s in segments
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or Path("outputs") / f"whisper-{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)

    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        path = out_dir / f"{uri}.json"
        if path.exists():
            continue
        audio = ami.load_audio(uri, args.data_dir)
        start = time.perf_counter()
        segments = transcribe(model, audio)
        elapsed = time.perf_counter() - start
        result = {
            "model": args.model,
            "compute_type": args.compute_type,
            "settings": SETTINGS,
            "audio_seconds": len(audio) / ami.SAMPLE_RATE,
            "elapsed_seconds": elapsed,
            "segments": segments,
        }
        path.write_text(json.dumps(result, indent=1))
        print(f"{uri}: {len(segments)} segments, {elapsed:.0f} s")


if __name__ == "__main__":
    main()
