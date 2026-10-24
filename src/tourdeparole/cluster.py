"""Cluster window embeddings into speakers with agglomerative clustering and write RTTMs.

    python -m tourdeparole.cluster --split dev --embeddings outputs/ecapa/vad-silero --tune
    python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-silero
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.cluster.hierarchy import fcluster, linkage

from tourdeparole import ami

# tuned on dev, see results/ahc_tuning_<vad>_dev.csv
THRESHOLD = 0.98
TUNE_THRESHOLDS = [round(t, 2) for t in np.arange(0.8, 1.21, 0.02)]
COMPONENTS = ["missed detection", "false alarm", "confusion"]


def dendrogram(embeddings: np.ndarray) -> np.ndarray:
    """Average linkage on cosine distances."""
    # embeddings share a large common component, without centering most windows look alike
    return linkage(embeddings - embeddings.mean(axis=0), method="average", metric="cosine")


def cluster(embeddings: np.ndarray, threshold: float | None = None, num_speakers: int | None = None) -> np.ndarray:
    """Cut the dendrogram at a distance threshold or at a number of clusters."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    tree = dendrogram(embeddings)
    if num_speakers is not None:
        return fcluster(tree, t=num_speakers, criterion="maxclust")
    return fcluster(tree, t=threshold, criterion="distance")


def diarization(uri: str, speech: np.ndarray, windows: np.ndarray, labels: np.ndarray) -> Annotation:
    """Label speech regions from the windows they contain, switching speaker halfway between window centers."""
    centers = windows.mean(axis=1)
    annotation = Annotation(uri=uri)
    for start, end in speech:
        inside = np.flatnonzero((windows[:, 0] >= start) & (windows[:, 1] <= end))
        if len(inside) == 0:
            nearest = np.argmin(np.abs(centers - (start + end) / 2))
            annotation[Segment(start, end)] = f"SPK{labels[nearest]:02d}"
            continue
        bounds = [start, *(centers[inside][:-1] + centers[inside][1:]) / 2, end]
        for i, k in enumerate(inside):
            annotation[Segment(bounds[i], bounds[i + 1])] = f"SPK{labels[k]:02d}"
    return annotation.support()


def run(uri: str, split: str, embeddings_dir: Path, threshold: float | None, known_speakers: bool, data_dir: Path) -> Annotation:
    data = np.load(embeddings_dir / f"{uri}.npz")
    num_speakers = len(ami.reference(uri, split, data_dir).labels()) if known_speakers else None
    labels = cluster(data["embeddings"], threshold, num_speakers)
    return diarization(uri, data["speech"], data["windows"], labels)


def tune(split: str, mini: bool, embeddings_dir: Path, data_dir: Path) -> list[dict]:
    """DER without collar over the split for each threshold, and how often the number of speakers is right."""
    metrics = {t: DiarizationErrorRate() for t in TUNE_THRESHOLDS}
    right = dict.fromkeys(TUNE_THRESHOLDS, 0)
    uris = ami.meetings(split, mini, data_dir)
    for uri in uris:
        data = np.load(embeddings_dir / f"{uri}.npz")
        reference, uem = ami.reference(uri, split, data_dir), ami.uem(uri, split, data_dir)
        tree = dendrogram(data["embeddings"])
        for t in TUNE_THRESHOLDS:
            hypothesis = diarization(uri, data["speech"], data["windows"], fcluster(tree, t=t, criterion="distance"))
            metrics[t](reference, hypothesis, uem=uem)
            right[t] += len(hypothesis.labels()) == len(reference.labels())

    rows = []
    for t, metric in metrics.items():
        total = metric.accumulated_["total"]
        rows.append(
            {"threshold": t, "der": round(100 * abs(metric), 2)}
            | {name.replace(" ", "_"): round(100 * metric.accumulated_[name] / total, 2) for name in COMPONENTS}
            | {"speaker_count_right": f"{right[t]}/{len(uris)}"}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--embeddings", type=Path, required=True, help="directory written by tourdeparole.embed")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="cosine distance at which clustering stops")
    parser.add_argument("--known-speakers", action="store_true", help="use the reference number of speakers instead")
    parser.add_argument("--tune", action="store_true", help="sweep the threshold and score DER instead")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, help="default: outputs/ecapa-ahc-<vad>")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    vad = args.embeddings.name.removeprefix("vad-")

    if args.tune:
        rows = tune(args.split, args.mini, args.embeddings, args.data_dir)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        name = f"ahc_tuning_{vad}_{args.split}{'_mini' if args.mini else ''}.csv"
        with open(args.results_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        best = min(rows, key=lambda r: r["der"])
        print(f"best: threshold {best['threshold']}, DER {best['der']:.2f}%, speaker count right {best['speaker_count_right']}")
        return

    out_dir = args.out_dir or Path("outputs") / f"ecapa-ahc-{vad}{'-known-speakers' if args.known_speakers else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        annotation = run(uri, args.split, args.embeddings, args.threshold, args.known_speakers, args.data_dir)
        with open(out_dir / f"{uri}.rttm", "w") as f:
            annotation.write_rttm(f)
        print(f"{uri}: {len(annotation.labels())} speakers")


if __name__ == "__main__":
    main()
