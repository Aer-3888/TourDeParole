"""Extract ECAPA-TDNN speaker embeddings on sliding windows over the detected speech.

    python -m tourdeparole.embed --split dev --vad outputs/vad-silero
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from tourdeparole import ami

MODEL = "speechbrain/spkrec-ecapa-voxceleb"
WINDOW = 1.5
HOP = 0.75
MIN_REGION = 0.25
BATCH_SIZE = 128


def windows(speech: Timeline) -> list[Segment]:
    """Windows of WINDOW seconds every HOP seconds, the last one ending with its region.

    A region shorter than WINDOW is one window, a region shorter than MIN_REGION gets none."""
    result = []
    for region in speech:
        if region.duration < MIN_REGION:
            continue
        if region.duration <= WINDOW:
            result.append(region)
            continue
        starts = np.arange(region.start, region.end - WINDOW, HOP)
        result += [Segment(s, s + WINDOW) for s in starts] + [Segment(region.end - WINDOW, region.end)]
    return result


@torch.inference_mode()
def embed(model: EncoderClassifier, audio: np.ndarray, segments: list[Segment]) -> np.ndarray:
    result = []
    for i in range(0, len(segments), BATCH_SIZE):
        chunks = [
            torch.from_numpy(audio[round(s.start * ami.SAMPLE_RATE) : round(s.end * ami.SAMPLE_RATE)])
            for s in segments[i : i + BATCH_SIZE]
        ]
        lengths = torch.tensor([len(c) for c in chunks], dtype=torch.float32)
        waveforms = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
        embeddings = model.encode_batch(waveforms.to(model.device), lengths / waveforms.shape[1])
        result.append(embeddings.squeeze(1).cpu().numpy())
    return np.concatenate(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--vad", type=Path, required=True, help="directory of speech regions as <meeting>.rttm")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, help="default: outputs/ecapa/<vad directory name>")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("outputs") / "ecapa" / args.vad.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = EncoderClassifier.from_hparams(
        source=MODEL, run_opts={"device": "cuda:0"}, local_strategy=LocalStrategy.NO_LINK
    )

    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        path = out_dir / f"{uri}.npz"
        if path.exists():
            continue
        speech = load_rttm(args.vad / f"{uri}.rttm")[uri].get_timeline().support()
        segments = windows(speech)
        start = time.perf_counter()
        embeddings = embed(model, ami.load_audio(uri, args.data_dir), segments)
        elapsed = time.perf_counter() - start
        np.savez(
            path,
            embeddings=embeddings,
            windows=np.array([[s.start, s.end] for s in segments]),
            speech=np.array([[s.start, s.end] for s in speech]),
            elapsed_seconds=elapsed,
        )
        print(f"{uri}: {len(segments)} windows, {elapsed:.0f} s")


if __name__ == "__main__":
    main()
