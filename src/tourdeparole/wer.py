"""Score Whisper transcripts against the AMI references with WER, ignoring speakers.

    python -m tourdeparole.wer --split test --hyp outputs/whisper-large-v3
"""

import argparse
import csv
import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from tourdeparole import ami, references

normalize = EnglishTextNormalizer()


def reference_text(uri: str, data_dir: Path) -> str:
    """Utterances of all speakers, concatenated in order of start time."""
    return normalize(" ".join(u["words"] for u in references.load(uri, data_dir)))


def hypothesis_text(path: Path) -> str:
    segments = json.loads(path.read_text())["segments"]
    return normalize(" ".join(s["text"] for s in segments))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=ami.SPLITS, default="test")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--hyp", type=Path, required=True, help="directory of transcribe outputs")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = []
    for uri in ami.meetings(args.split, args.mini, args.data_dir):
        out = jiwer.process_words(reference_text(uri, args.data_dir), hypothesis_text(args.hyp / f"{uri}.json"))
        n_ref = out.hits + out.substitutions + out.deletions
        rows.append(
            {
                "meeting": uri,
                "ref_words": n_ref,
                "substitutions": out.substitutions,
                "deletions": out.deletions,
                "insertions": out.insertions,
            }
        )
    total = {"meeting": "TOTAL"} | {k: sum(r[k] for r in rows) for k in rows[0] if k != "meeting"}
    rows.append(total)
    for r in rows:
        errors = r["substitutions"] + r["deletions"] + r["insertions"]
        r["wer"] = round(100 * errors / r["ref_words"], 2)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    name = f"wer_{args.hyp.name}_{args.split}{'_mini' if args.mini else ''}.csv"
    with open(args.results_dir / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['meeting']:>8}  WER {r['wer']:6.2f}%  ({r['ref_words']} words)")


if __name__ == "__main__":
    main()
