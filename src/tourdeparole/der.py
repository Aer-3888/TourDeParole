"""Score diarization RTTMs against the AMI references with DER.

    python -m tourdeparole.der --split test --hyp outputs/pyannote-community-1
"""

import argparse
from pathlib import Path

from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from tourdeparole import ami, tables

# pyannote.metrics collars are centered on each boundary, so 0.5 means 0.25 s on each side
SETTINGS = {
    "full": {"collar": 0.0, "skip_overlap": False},
    "fair": {"collar": 0.5, "skip_overlap": False},
    "forgiving": {"collar": 0.5, "skip_overlap": True},
}
COMPONENTS = ["missed detection", "false alarm", "confusion"]


def score(split: str, mini: bool, hyp_dir: Path, data_dir: Path) -> list[dict]:
    rows = []
    for setting, params in SETTINGS.items():
        metric = DiarizationErrorRate(**params)
        totals = dict.fromkeys(["total", *COMPONENTS], 0.0)
        for uri in ami.meetings(split, mini, data_dir):
            hypothesis = load_rttm(hyp_dir / f"{uri}.rttm").get(uri, Annotation(uri=uri))
            details = metric(ami.reference(uri, split, data_dir), hypothesis, uem=ami.uem(uri, split, data_dir), detailed=True)
            rows.append(row(setting, uri, details))
            for key in totals:
                totals[key] += details[key]
        rows.append(row(setting, "TOTAL", totals))
    return rows


def row(setting: str, uri: str, details: dict) -> dict:
    total = details["total"]
    result = {"setting": setting, "meeting": uri, "speech_seconds": round(total, 2)}
    result |= {name.replace(" ", "_"): round(100 * details[name] / total, 2) for name in COMPONENTS}
    result["der"] = round(100 * sum(details[name] for name in COMPONENTS) / total, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--hyp", type=Path, required=True, help="directory of <meeting>.rttm files")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = score(args.split, args.mini, args.hyp, args.data_dir)
    name = f"der_{args.hyp.name}_{args.split}{'_mini' if args.mini else ''}.csv"
    tables.write_csv(args.results_dir / name, rows)
    for r in rows:
        if r["meeting"] == "TOTAL":
            print(
                f"{r['setting']:>9}  DER {r['der']:5.2f}%  missed {r['missed_detection']:5.2f}%  "
                f"false alarm {r['false_alarm']:5.2f}%  confusion {r['confusion']:5.2f}%"
            )


if __name__ == "__main__":
    main()
