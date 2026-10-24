"""Error analysis: overlapped speech, speaker counts and short turns.

    python -m tourdeparole.analysis data --split test
    python -m tourdeparole.analysis diarization --split test --hyp outputs/pyannote-community-1
    python -m tourdeparole.analysis transcription --split test --wer results/wer_whisper-large-v3_test.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from tourdeparole import ami

COMPONENTS = ["missed detection", "false alarm", "confusion"]
TURN_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float("inf"))]
SPEAKER_COUNTS = {"speakers", "ref_speakers", "hyp_speakers"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def total_row(rows: list[dict]) -> dict:
    return {"meeting": "TOTAL"} | {
        k: "" if k in SPEAKER_COUNTS else round(sum(r[k] for r in rows), 2) for k in rows[0] if k != "meeting"
    }


def overlap_share(reference: Annotation) -> float:
    """Share of speech time where at least two speakers talk at once."""
    return reference.get_overlap().duration() / reference.get_timeline().support().duration()


def data_stats(split: str, mini: bool, data_dir: Path) -> list[dict]:
    rows = []
    for uri in ami.meetings(split, mini, data_dir):
        reference = ami.reference(uri, split, data_dir)
        rows.append(
            {
                "meeting": uri,
                "seconds": round(ami.uem(uri, split, data_dir).duration(), 2),
                "speakers": len(reference.labels()),
                "speech_seconds": round(reference.get_timeline().support().duration(), 2),
                "overlap_seconds": round(reference.get_overlap().duration(), 2),
            }
        )
    rows.append(total_row(rows))
    for r in rows:
        r["overlap_share"] = round(100 * r["overlap_seconds"] / r["speech_seconds"], 2)
    return rows


def load_hypothesis(hyp_dir: Path, uri: str) -> Annotation:
    return load_rttm(hyp_dir / f"{uri}.rttm").get(uri, Annotation(uri=uri))


def region_errors(split: str, mini: bool, hyp_dir: Path, data_dir: Path) -> list[dict]:
    """DER components in seconds, split between overlapped speech and the rest of the meeting."""
    metric = DiarizationErrorRate()
    rows = []
    for uri in ami.meetings(split, mini, data_dir):
        reference, uem = ami.reference(uri, split, data_dir), ami.uem(uri, split, data_dir)
        hypothesis = load_hypothesis(hyp_dir, uri)
        overlap = reference.get_overlap().crop(uem)
        mapping = metric.optimal_mapping(reference, hypothesis, uem=uem)
        unmatched = [label for label in hypothesis.labels() if label not in mapping]
        row = {
            "meeting": uri,
            "ref_speakers": len(reference.labels()),
            "hyp_speakers": len(hypothesis.labels()),
            "unmatched_hyp_seconds": round(sum(hypothesis.label_duration(label) for label in unmatched), 2),
        }
        for region, timeline in [("overlap", overlap), ("single", uem.extrude(overlap))]:
            details = metric(reference, hypothesis, uem=timeline, detailed=True)
            row |= {f"{name.replace(' ', '_')}_{region}": round(details[name], 2) for name in COMPONENTS}
        rows.append(row)
    rows.append(total_row(rows))
    return rows


def turn_errors(split: str, mini: bool, hyp_dir: Path, data_dir: Path) -> list[dict]:
    """Share of each reference turn given to the right speaker, missed, or given to another speaker, by turn length."""
    metric = DiarizationErrorRate()
    totals = {bounds: dict.fromkeys(["turns", "seconds", "correct", "missed"], 0.0) for bounds in TURN_BINS}
    for uri in ami.meetings(split, mini, data_dir):
        reference, uem = ami.reference(uri, split, data_dir), ami.uem(uri, split, data_dir)
        hypothesis = load_hypothesis(hyp_dir, uri)
        hypothesis = hypothesis.rename_labels(metric.optimal_mapping(reference, hypothesis, uem=uem))
        detected = hypothesis.get_timeline().support()
        for turn, _, speaker in reference.itertracks(yield_label=True):
            bounds = next(b for b in TURN_BINS if b[0] <= turn.duration < b[1])
            window = Timeline([turn])
            totals[bounds]["turns"] += 1
            totals[bounds]["seconds"] += turn.duration
            totals[bounds]["correct"] += hypothesis.label_timeline(speaker).crop(window).duration()
            totals[bounds]["missed"] += turn.duration - detected.crop(window).duration()

    rows = []
    for (low, high), t in totals.items():
        rows.append(
            {
                "turn_length": f"{low:g}-{high:g} s" if high != float("inf") else f">{low:g} s",
                "turns": int(t["turns"]),
                "seconds": round(t["seconds"], 2),
                "correct": round(100 * t["correct"] / t["seconds"], 2),
                "missed": round(100 * t["missed"] / t["seconds"], 2),
                "other_speaker": round(100 * (t["seconds"] - t["correct"] - t["missed"]) / t["seconds"], 2),
            }
        )
    return rows


def suffix(args: argparse.Namespace) -> str:
    return f"{args.split}{'_mini' if args.mini else ''}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="duration, speakers and overlapped speech of the references")
    diarization = commands.add_parser("diarization", help="errors in overlapped speech, speaker counts, short turns")
    diarization.add_argument("--hyp", type=Path, required=True, help="directory of <meeting>.rttm files")
    transcription = commands.add_parser("transcription", help="correlation of WER with overlapped speech")
    transcription.add_argument("--wer", type=Path, required=True, help="CSV written by tourdeparole.wer")
    for command in (data, diarization, transcription):
        command.add_argument("--split", choices=ami.SPLITS, default="test")
        command.add_argument("--mini", action="store_true")
        command.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
        command.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.command == "data":
        rows = data_stats(args.split, args.mini, args.data_dir)
        write_csv(args.results_dir / f"data_{suffix(args)}.csv", rows)
        total = rows[-1]
        print(
            f"{len(rows) - 1} meetings, {total['seconds'] / 3600:.2f} h, "
            f"{total['overlap_share']:.1f}% of speech overlapped"
        )

    elif args.command == "diarization":
        rows = region_errors(args.split, args.mini, args.hyp, args.data_dir)
        write_csv(args.results_dir / f"regions_{args.hyp.name}_{suffix(args)}.csv", rows)
        total = rows[-1]
        missed = total["missed_detection_overlap"] + total["missed_detection_single"]
        right = sum(r["ref_speakers"] == r["hyp_speakers"] for r in rows[:-1])
        print(f"missed speech in overlap: {total['missed_detection_overlap']:.0f} of {missed:.0f} s")
        print(f"speaker count right in {right} of {len(rows) - 1} meetings")

        rows = turn_errors(args.split, args.mini, args.hyp, args.data_dir)
        write_csv(args.results_dir / f"turns_{args.hyp.name}_{suffix(args)}.csv", rows)
        for r in rows:
            print(
                f"{r['turn_length']:>7}  {r['turns']:5d} turns  correct {r['correct']:5.1f}%  "
                f"missed {r['missed']:5.1f}%  other speaker {r['other_speaker']:5.1f}%"
            )

    else:
        with open(args.wer, newline="") as f:
            wer = {r["meeting"]: float(r["wer"]) for r in csv.DictReader(f) if r["meeting"] != "TOTAL"}
        shares = [overlap_share(ami.reference(uri, args.split, args.data_dir)) for uri in wer]
        r = np.corrcoef(shares, list(wer.values()))[0, 1]
        print(f"Pearson r between WER and overlapped speech: {r:.2f} over {len(wer)} meetings")


if __name__ == "__main__":
    main()
