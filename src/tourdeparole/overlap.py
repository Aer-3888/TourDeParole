"""Detect overlapped speech with the pyannote segmentation model.

    python -m tourdeparole.overlap detect --split test
"""

import argparse
import csv
import os
from pathlib import Path

# pyannote sends usage telemetry unless this is set before import
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Timeline
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure

from tourdeparole import ami

PIPELINE = "pyannote/speaker-diarization-community-1"
SCORES_DIR = Path("outputs") / "overlap-scores"
THRESHOLD = 0.5


def overlap_inference() -> Inference:
    """Probability that at least two speakers are active, from the powerset outputs of the segmentation model.

    The model classifies each frame of a 10 s chunk into no speaker, one of three local speakers, or a pair."""
    model = Model.from_pretrained(PIPELINE, subfolder="segmentation")
    specifications = model.specifications
    powerset = Powerset(len(specifications.classes), specifications.powerset_max_classes)
    pairs = (powerset.mapping.sum(dim=1) >= 2).numpy()
    return Inference(
        model,
        step=0.1 * specifications.duration,
        skip_conversion=True,
        pre_aggregation_hook=lambda log_probs: np.exp(log_probs)[..., pairs].sum(axis=-1, keepdims=True),
        device=torch.device("cuda"),
    )


def save_scores(path: Path, scores: SlidingWindowFeature) -> None:
    window = scores.sliding_window
    np.savez(path, data=scores.data, start=window.start, duration=window.duration, step=window.step)


def load_scores(uri: str, scores_dir: Path = SCORES_DIR) -> SlidingWindowFeature:
    saved = np.load(scores_dir / f"{uri}.npz")
    window = SlidingWindow(start=float(saved["start"]), duration=float(saved["duration"]), step=float(saved["step"]))
    return SlidingWindowFeature(saved["data"], window)


def detect(uri: str, threshold: float, scores_dir: Path = SCORES_DIR) -> Timeline:
    return Binarize(onset=threshold)(load_scores(uri, scores_dir)).get_timeline().support()


def reference_overlap(uri: str, split: str, data_dir: Path) -> Annotation:
    overlap = Annotation(uri=uri)
    for segment in ami.reference(uri, split, data_dir).get_overlap():
        overlap[segment] = "overlap"
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    detection = commands.add_parser("detect", help="overlap scores per meeting, then precision and recall")
    detection.add_argument("--threshold", type=float, default=THRESHOLD)
    for command in (detection,):
        command.add_argument("--split", choices=ami.SPLITS, default="test")
        command.add_argument("--mini", action="store_true")
        command.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
        command.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    suffix = f"{args.split}{'_mini' if args.mini else ''}"

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    inference = None
    metric = DetectionPrecisionRecallFMeasure()
    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        path = SCORES_DIR / f"{uri}.npz"
        if not path.exists():
            inference = inference or overlap_inference()
            audio = ami.load_audio(uri, args.data_dir)
            save_scores(path, inference({"waveform": torch.from_numpy(audio)[None], "sample_rate": ami.SAMPLE_RATE}))
        detected = Annotation(uri=uri)
        for segment in detect(uri, args.threshold):
            detected[segment] = "overlap"
        metric(reference_overlap(uri, args.split, args.data_dir), detected, uem=ami.uem(uri, args.split, args.data_dir))

    precision, recall, _ = metric.compute_metrics()
    row = {"threshold": args.threshold, "precision": round(100 * precision, 2), "recall": round(100 * recall, 2)}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / f"overlap_detection_{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"overlap detection at {args.threshold}: precision {row['precision']:.2f}%  recall {row['recall']:.2f}%")


if __name__ == "__main__":
    main()
