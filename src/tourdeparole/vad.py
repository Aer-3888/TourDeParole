"""Detect speech with Silero VAD, or take it from the reference (oracle), and score it.

    python -m tourdeparole.vad --split dev --tune
    python -m tourdeparole.vad --split test --method silero
    python -m tourdeparole.vad --split test --method oracle
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from faster_whisper.vad import VadOptions, get_speech_timestamps
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate

from tourdeparole import ami, tables

# Silero defaults for minimum speech and silence, faster-whisper's are meant for cutting Whisper chunks
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 100
# tuned on dev, see results/vad_tuning_dev.csv
THRESHOLD = 0.3
PAD_MS = 200
TUNE_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)
TUNE_PADS_MS = (30, 100, 200, 300, 400)


def silero(uri: str, audio, threshold: float, pad_ms: int) -> Annotation:
    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=pad_ms,
    )
    speech = Annotation(uri=uri)
    for chunk in get_speech_timestamps(audio, options, sampling_rate=ami.SAMPLE_RATE):
        speech[Segment(chunk["start"] / ami.SAMPLE_RATE, chunk["end"] / ami.SAMPLE_RATE)] = "speech"
    return speech.support()


def oracle(uri: str, split: str, data_dir: Path) -> Annotation:
    speech = Annotation(uri=uri)
    for segment in ami.reference(uri, split, data_dir).get_timeline().support():
        speech[segment] = "speech"
    return speech


def detection_errors(uri: str, split: str, speech: Annotation, data_dir: Path) -> dict:
    metric = DetectionErrorRate()
    details = metric(ami.reference(uri, split, data_dir), speech, uem=ami.uem(uri, split, data_dir), detailed=True)
    return {"speech_seconds": details["total"], "missed_seconds": details["miss"], "false_alarm_seconds": details["false alarm"]}


def tune_meeting(uri: str, split: str, data_dir: Path) -> list[dict]:
    audio = ami.load_audio(uri, data_dir)
    rows = []
    for threshold, pad_ms in itertools.product(TUNE_THRESHOLDS, TUNE_PADS_MS):
        speech = silero(uri, audio, threshold, pad_ms)
        rows.append({"threshold": threshold, "pad_ms": pad_ms} | detection_errors(uri, split, speech, data_dir))
    return rows


def tune(split: str, mini: bool, data_dir: Path, workers: int) -> list[dict]:
    uris = ami.meetings(split, mini, data_dir)
    totals = {}
    with ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(tune_meeting, uris, itertools.repeat(split), itertools.repeat(data_dir)):
            for r in rows:
                key = (r["threshold"], r["pad_ms"])
                total = totals.setdefault(key, dict.fromkeys(["speech_seconds", "missed_seconds", "false_alarm_seconds"], 0.0))
                for k in total:
                    total[k] += r[k]
    return [{"threshold": t, "pad_ms": p} | percentages(total) for (t, p), total in totals.items()]


def percentages(errors: dict) -> dict:
    total = errors["speech_seconds"]
    missed = 100 * errors["missed_seconds"] / total
    false_alarm = 100 * errors["false_alarm_seconds"] / total
    return {
        "speech_seconds": round(total, 2),
        "missed": round(missed, 2),
        "false_alarm": round(false_alarm, 2),
        "error": round(missed + false_alarm, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--method", choices=["silero", "oracle"], default="silero")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--pad-ms", type=int, default=PAD_MS)
    parser.add_argument("--tune", action="store_true", help="sweep Silero threshold and padding instead")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    suffix = f"{args.split}{'_mini' if args.mini else ''}"

    if args.tune:
        rows = tune(args.split, args.mini, args.data_dir, args.workers)
        tables.write_csv(args.results_dir / f"vad_tuning_{suffix}.csv", rows)
        best = min(rows, key=lambda r: r["error"])
        print(f"best: threshold {best['threshold']}, padding {best['pad_ms']} ms, error {best['error']:.2f}%")
        return

    out_dir = Path("outputs") / f"vad-{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        path = out_dir / f"{uri}.rttm"
        if path.exists():
            speech = load_rttm(path)[uri]
        else:
            if args.method == "silero":
                speech = silero(uri, ami.load_audio(uri, args.data_dir), args.threshold, args.pad_ms)
            else:
                speech = oracle(uri, args.split, args.data_dir)
            with open(path, "w") as f:
                speech.write_rttm(f)
        if args.method == "silero":
            rows.append({"meeting": uri} | detection_errors(uri, args.split, speech, args.data_dir))

    if args.method == "oracle":
        return
    total = {k: sum(r[k] for r in rows) for k in rows[0] if k != "meeting"}
    rows = [{"meeting": r["meeting"]} | percentages(r) for r in rows] + [{"meeting": "TOTAL"} | percentages(total)]
    tables.write_csv(args.results_dir / f"vad_{args.method}_{suffix}.csv", rows)
    print(f"missed {rows[-1]['missed']:.2f}%  false alarm {rows[-1]['false_alarm']:.2f}%")


if __name__ == "__main__":
    main()
