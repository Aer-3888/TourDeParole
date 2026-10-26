"""Figures for the README: DER breakdown per system, and a speaker timeline of a meeting excerpt.

    python -m tourdeparole.plot der
    python -m tourdeparole.plot timeline --meeting EN2002d --start 2055 --duration 60
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from tourdeparole import ami

SYSTEMS = {
    "pyannote community-1": Path("outputs/pyannote-community-1"),
    "Clustering baseline": Path("outputs/ecapa-ahc-silero"),
    "+ overlap handling": Path("outputs/ecapa-ahc-silero-overlap"),
}
DER_LABELS = {
    "pyannote community-1": "pyannote community-1\n(pretrained)",
    "Clustering baseline": "Clustering baseline",
    "+ overlap handling": "Baseline\n+ overlap handling",
}
COMPONENTS = {
    "missed_detection": "Missed speech (a speaker talks but is not detected)",
    "false_alarm": "False alarm (speech detected in silence)",
    "confusion": "Wrong speaker",
}
# GitHub page backgrounds, so each version blends into its theme
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "muted": "#52514e", "rule": "#d6d5d0",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#0d1117", "ink": "#f0efec", "muted": "#c3c2b7", "rule": "#3d3d3a",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
# Okabe-Ito, readable with color vision deficiencies
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
UNMATCHED = "#999999"


def der_totals(results_dir: Path, split: str) -> dict[str, dict[str, float]]:
    totals = {}
    for name, hyp_dir in SYSTEMS.items():
        with open(results_dir / f"der_{hyp_dir.name}_{split}.csv", newline="") as f:
            row = next(r for r in csv.DictReader(f) if r["setting"] == "full" and r["meeting"] == "TOTAL")
        totals[name] = {key: float(row[key]) for key in [*COMPONENTS, "der"]}
    return totals


def draw_der(totals: dict[str, dict[str, float]], theme: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.4), facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    y = list(range(len(totals)))[::-1]
    left = [0.0] * len(totals)
    for (key, label), color in zip(COMPONENTS.items(), theme["series"]):
        values = [t[key] for t in totals.values()]
        ax.barh(y, values, left=left, height=0.58, color=color, edgecolor=theme["surface"], linewidth=2, label=label)
        for yi, start, value in zip(y, left, values):
            if value >= 3:
                ax.text(start + value / 2, yi, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        left = [a + b for a, b in zip(left, values)]
    for yi, t in zip(y, totals.values()):
        ax.text(t["der"] + 0.4, yi, f"{t['der']:.1f}%", va="center", color=theme["ink"], fontsize=10, fontweight="bold")

    ax.set_yticks(y, [DER_LABELS[name] for name in totals], color=theme["ink"], fontsize=9)
    ax.set_xlim(0, max(t["der"] for t in totals.values()) * 1.15)
    ax.set_xlabel("Diarization error rate: % of speech time attributed wrongly (lower is better)", color=theme["muted"], fontsize=8.5)
    ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["rule"])
    fig.suptitle(title, x=0.02, y=1.0, ha="left", fontsize=11, fontweight="bold", color=theme["ink"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False, fontsize=8,
              labelcolor=theme["ink"], handlelength=1.2, columnspacing=1.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def mapped(hypothesis: Annotation, reference: Annotation, uem) -> Annotation:
    """Rename hypothesis speakers to the reference speakers they match over the whole meeting."""
    mapping = DiarizationErrorRate().optimal_mapping(reference, hypothesis, uem=uem)
    return hypothesis.rename_labels(mapping={label: mapping.get(label, f"extra {label}") for label in hypothesis.labels()})


def draw_timeline(rows: dict[str, Annotation], reference: Annotation, window: Segment, title: str, path: Path) -> None:
    speakers = reference.labels()
    colors = dict(zip(speakers, COLORS))
    lane = 1.0
    gap = 0.8
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.32 * len(rows) * (len(speakers) + gap)))

    for segment in reference.get_overlap():
        ax.axvspan(segment.start, segment.end, color="#000000", alpha=0.08, linewidth=0)

    ticks, labels = [], []
    top = 0.0
    for n, (name, annotation) in enumerate(rows.items()):
        if n:
            ax.axhline(top + (gap + lane) / 2, color="#cccccc", linewidth=0.6)
        for i, speaker in enumerate(speakers):
            y = top - i * lane
            for segment in annotation.label_timeline(speaker, copy=False) if speaker in annotation.labels() else []:
                ax.broken_barh([(segment.start, segment.duration)], (y - 0.4, 0.8), color=colors[speaker])
        for label in annotation.labels():
            if label not in colors:
                y = top - (len(speakers) - 1) * lane
                for segment in annotation.label_timeline(label, copy=False):
                    ax.broken_barh([(segment.start, segment.duration)], (y - 0.15, 0.3), color=UNMATCHED)
        ticks.append(top - (len(speakers) - 1) * lane / 2)
        labels.append(name)
        top -= len(speakers) * lane + gap

    ax.set_yticks(ticks, labels)
    ax.set_xlim(window.start, window.end)
    ax.set_ylim(top + gap - 0.5, 0.6)
    ax.xaxis.set_major_formatter(lambda t, _: f"{int(t // 60)}:{int(t % 60):02d}")
    ax.set_xlabel("Time in the meeting (min:s)")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Patch(color=colors[s], label=f"Speaker {s}") for s in speakers]
    handles.append(Patch(color="#000000", alpha=0.08, label="Overlapped speech (reference)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(handles), frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    der = commands.add_parser("der", help="DER breakdown per system, light and dark versions")
    der.add_argument("--split", choices=ami.SPLITS, default="test")
    der.add_argument("--results-dir", type=Path, default=Path("results"))
    der.add_argument("--out-dir", type=Path, default=Path("figures"))

    timeline = commands.add_parser("timeline", help="speaker timeline of a meeting excerpt")
    timeline.add_argument("--meeting", required=True)
    timeline.add_argument("--split", choices=ami.SPLITS, default="test")
    timeline.add_argument("--start", type=float, required=True, help="seconds")
    timeline.add_argument("--duration", type=float, default=60.0, help="seconds")
    timeline.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    timeline.add_argument("--out", type=Path, default=None, help="default: figures/timeline_<meeting>.png")
    args = parser.parse_args()

    if args.command == "der":
        totals = der_totals(args.results_dir, args.split)
        for name, theme in THEMES.items():
            out = args.out_dir / f"der_{args.split}_{name}.png"
            draw_der(totals, theme, f"Who spoke when? DER on the AMI {args.split} set by error type", out)
            print(f"wrote {out}")
        return

    uri = args.meeting
    window = Segment(args.start, args.start + args.duration)
    reference = ami.reference(uri, args.split, args.data_dir)
    uem = ami.uem(uri, args.split, args.data_dir)
    rows = {"Reference": reference.crop(window)}
    for name, hyp_dir in SYSTEMS.items():
        hypothesis = load_rttm(hyp_dir / f"{uri}.rttm").get(uri, Annotation(uri=uri))
        rows[name] = mapped(hypothesis, reference, uem).crop(window)

    out = args.out or Path("figures") / f"timeline_{uri}.png"
    draw_timeline(rows, reference.crop(window), window, f"AMI {uri}, {args.duration:.0f} s excerpt", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
