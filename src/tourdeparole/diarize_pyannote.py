"""Run the pretrained pyannote speaker diarization pipeline on AMI meetings.

Needs a Hugging Face token (`hf auth login`) with the model conditions accepted:

    python -m tourdeparole.diarize_pyannote --split test
"""

import argparse
import json
import os
import time
from pathlib import Path

# pyannote sends usage telemetry unless this is set before import
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import torch
from pyannote.audio import Pipeline

from tourdeparole import ami

PIPELINE = "pyannote/speaker-diarization-community-1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "pyannote-community-1")
    args = parser.parse_args()

    (args.out_dir / "exclusive").mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline.from_pretrained(PIPELINE).to(torch.device("cuda"))

    timings = {}
    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        path = args.out_dir / f"{uri}.rttm"
        if path.exists():
            continue
        audio = ami.load_audio(uri, args.data_dir)
        waveform = {"waveform": torch.from_numpy(audio)[None], "sample_rate": ami.SAMPLE_RATE}
        start = time.perf_counter()
        output = pipeline(waveform)
        timings[uri] = time.perf_counter() - start
        for annotation, target in [
            (output.speaker_diarization, path),
            (output.exclusive_speaker_diarization, args.out_dir / "exclusive" / f"{uri}.rttm"),
        ]:
            annotation.uri = uri
            with open(target, "w") as f:
                annotation.write_rttm(f)
        print(f"{uri}: {len(output.speaker_diarization.labels())} speakers, {timings[uri]:.0f} s")

    timings_path = args.out_dir / "timings.json"
    previous = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    timings_path.write_text(json.dumps(previous | timings, indent=1))


if __name__ == "__main__":
    main()
