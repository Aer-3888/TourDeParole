"""Detect overlapped speech with the pyannote segmentation model and add a second speaker there.

    python -m tourdeparole.overlap detect --split test
    python -m tourdeparole.overlap tune --split dev --hyp outputs/ecapa-ahc-silero --embeddings outputs/ecapa/vad-silero
    python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-silero
"""

import argparse
import itertools
import os
from collections.abc import Callable
from pathlib import Path

# pyannote sends usage telemetry unless this is set before import
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.metrics.diarization import DiarizationErrorRate

from tourdeparole import ami, cluster, tables

PIPELINE = "pyannote/speaker-diarization-community-1"
SCORES_DIR = Path("outputs") / "overlap-scores"
# tuned on dev, see results/overlap_tuning_<system>_dev.csv
THRESHOLD = 0.6
TUNE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
RULES = ("temporal", "embedding")
COMPONENTS = ["missed detection", "false alarm", "confusion"]


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


def nearest_in_time(i: int, starts: np.ndarray, ends: np.ndarray, labels: list[str]) -> str | None:
    """Speaker of the closest turn of another speaker, before or after turn i."""
    before = next((j for j in range(i - 1, -1, -1) if labels[j] != labels[i]), None)
    after = next((j for j in range(i + 1, len(labels)) if labels[j] != labels[i]), None)
    candidates = []
    if before is not None:
        candidates.append((starts[i] - ends[before], labels[before]))
    if after is not None:
        candidates.append((starts[after] - ends[i], labels[after]))
    return min(candidates)[1] if candidates else None


def nearest_in_embedding(piece: Segment, first: str, embeddings: np.ndarray, windows: np.ndarray, centroids: dict) -> str | None:
    """Other speaker whose centroid is closest to the mean embedding of the windows over the piece."""
    inside = (windows[:, 0] < piece.end) & (windows[:, 1] > piece.start)
    if not inside.any():
        return None
    x = embeddings[inside].mean(axis=0)
    similarity = {label: x @ c / (np.linalg.norm(x) * np.linalg.norm(c)) for label, c in centroids.items() if label != first}
    return max(similarity, key=similarity.get) if similarity else None


def add_second_speakers(hypothesis: Annotation, overlap: Timeline, rule: Callable[[int, Segment], str | None]) -> Annotation:
    """Give each turn a second speaker where it meets overlapped speech. Turns must not overlap each other."""
    tracks = sorted((segment.start, segment.end, label) for segment, _, label in hypothesis.itertracks(yield_label=True))
    regions = np.array([[s.start, s.end] for s in overlap]).reshape(-1, 2)
    result = hypothesis.copy()
    for i, (start, end, _) in enumerate(tracks):
        for k in range(np.searchsorted(regions[:, 1], start, side="right"), len(regions)):
            if regions[k, 0] >= end:
                break
            piece = Segment(max(start, regions[k, 0]), min(end, regions[k, 1]))
            second = rule(i, piece)
            if second is not None:
                result[piece, "overlap"] = second
    return result


def second_speakers(hypothesis: Annotation, overlap: Timeline, rule: str, embeddings_path: Path | None = None) -> Annotation:
    tracks = sorted((segment.start, segment.end, label) for segment, _, label in hypothesis.itertracks(yield_label=True))
    starts, ends, labels = np.array([t[0] for t in tracks]), np.array([t[1] for t in tracks]), [t[2] for t in tracks]
    if rule == "temporal":
        return add_second_speakers(hypothesis, overlap, lambda i, piece: nearest_in_time(i, starts, ends, labels))

    # same clusters and labels as tourdeparole.cluster at its default threshold
    data = np.load(embeddings_path)
    embeddings = data["embeddings"] - data["embeddings"].mean(axis=0)
    clusters = cluster.cluster(data["embeddings"], cluster.THRESHOLD)
    centroids = {f"SPK{k:02d}": embeddings[clusters == k].mean(axis=0) for k in np.unique(clusters)}
    return add_second_speakers(
        hypothesis, overlap, lambda i, piece: nearest_in_embedding(piece, labels[i], embeddings, data["windows"], centroids)
    )


def tune(split: str, mini: bool, hyp_dir: Path, embeddings_dir: Path, data_dir: Path) -> list[dict]:
    """DER without collar over the split for each rule and threshold, and with the reference overlapped speech."""
    settings = [(rule, t) for rule, t in itertools.product(RULES, TUNE_THRESHOLDS)] + [(rule, "reference") for rule in RULES]
    metrics = {setting: DiarizationErrorRate() for setting in [("none", "none"), *settings]}
    for uri in ami.meetings(split, mini, data_dir):
        reference, uem = ami.reference(uri, split, data_dir), ami.uem(uri, split, data_dir)
        hypothesis = load_rttm(hyp_dir / f"{uri}.rttm")[uri]
        metrics["none", "none"](reference, hypothesis, uem=uem)
        for rule, threshold in settings:
            overlap = reference.get_overlap() if threshold == "reference" else detect(uri, threshold)
            updated = second_speakers(hypothesis, overlap, rule, embeddings_dir / f"{uri}.npz")
            metrics[rule, threshold](reference, updated, uem=uem)

    rows = []
    for (rule, threshold), metric in metrics.items():
        total = metric.accumulated_["total"]
        rows.append(
            {"rule": rule, "threshold": threshold, "der": round(100 * abs(metric), 2)}
            | {name.replace(" ", "_"): round(100 * metric.accumulated_[name] / total, 2) for name in COMPONENTS}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    detection = commands.add_parser("detect", help="overlap scores per meeting, then precision and recall")
    detection.add_argument("--threshold", type=float, default=THRESHOLD)
    tuning = commands.add_parser("tune", help="sweep the rule and the detection threshold, scoring DER")
    tuning.add_argument("--hyp", type=Path, required=True, help="directory of <meeting>.rttm files")
    tuning.add_argument("--embeddings", type=Path, required=True, help="embeddings the hypothesis was clustered from")
    assignment = commands.add_parser("assign", help="add a second speaker in overlapped speech and write RTTMs")
    assignment.add_argument("--hyp", type=Path, required=True, help="directory of <meeting>.rttm files")
    assignment.add_argument("--rule", choices=RULES, default="temporal")
    assignment.add_argument("--embeddings", type=Path, help="needed by the embedding rule")
    assignment.add_argument("--threshold", type=float, default=THRESHOLD)
    assignment.add_argument("--reference", action="store_true", help="use the reference overlapped speech instead")
    for command in (detection, tuning, assignment):
        command.add_argument("--split", choices=ami.SPLITS, default="test")
        command.add_argument("--mini", action="store_true")
        command.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
        command.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    suffix = f"{args.split}{'_mini' if args.mini else ''}"

    if args.command == "detect":
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
        tables.write_csv(args.results_dir / f"overlap_detection_{suffix}.csv", [row])
        print(f"overlap detection at {args.threshold}: precision {row['precision']:.2f}%  recall {row['recall']:.2f}%")

    elif args.command == "tune":
        rows = tune(args.split, args.mini, args.hyp, args.embeddings, args.data_dir)
        tables.write_csv(args.results_dir / f"overlap_tuning_{args.hyp.name}_{suffix}.csv", rows)
        for r in rows:
            print(f"{r['rule']:>9} {r['threshold']!s:>9}  DER {r['der']:5.2f}%  missed {r['missed_detection']:5.2f}%  "
                  f"false alarm {r['false_alarm']:5.2f}%  confusion {r['confusion']:5.2f}%")

    else:
        if args.rule == "embedding" and args.embeddings is None:
            parser.error("the embedding rule needs --embeddings")
        name = f"{args.hyp.name}-{'reference-' if args.reference else ''}overlap{'-embedding' if args.rule == 'embedding' else ''}"
        out_dir = Path("outputs") / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for uri in ami.meetings(args.split, args.mini, args.data_dir):
            hypothesis = load_rttm(args.hyp / f"{uri}.rttm")[uri]
            overlap = ami.reference(uri, args.split, args.data_dir).get_overlap() if args.reference else detect(uri, args.threshold)
            embeddings = args.embeddings / f"{uri}.npz" if args.embeddings else None
            with open(out_dir / f"{uri}.rttm", "w") as f:
                second_speakers(hypothesis, overlap, args.rule, embeddings).write_rttm(f)
        print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
