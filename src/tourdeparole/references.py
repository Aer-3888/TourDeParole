"""Build speaker-attributed reference transcripts from the AMI manual annotations.

Writes one SegLST file per meeting, the JSON format read by meeteval:

    python -m tourdeparole.references --splits test dev
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from tourdeparole import ami

NITE = "{http://nite.sourceforge.net/}"
HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def references_dir(data_dir: Path = ami.DATA_DIR) -> Path:
    return data_dir / "references"


def speakers(uri: str, data_dir: Path) -> dict[str, str]:
    """Map NXT agent letters (A, B...) to the global speaker names used in the RTTMs."""
    path = ami.annotations_dir(data_dir) / "corpusResources" / "meetings.xml"
    meeting = ET.parse(path).getroot().find(f".//meeting[@observation='{uri}']")
    return {s.get("nxt_agent"): s.get("global_name") for s in meeting.iter("speaker")}


def utterances(uri: str, agent: str, speaker: str, data_dir: Path) -> list[dict]:
    annotations = ami.annotations_dir(data_dir)
    elements = list(ET.parse(annotations / "words" / f"{uri}.{agent}.words.xml").getroot())
    position = {e.get(f"{NITE}id"): i for i, e in enumerate(elements)}
    segments = ET.parse(annotations / "segments" / f"{uri}.{agent}.segments.xml").getroot()

    result = []
    for segment in segments.iter("segment"):
        words = []
        for child in segment.iter(f"{NITE}child"):
            first, last = HREF.search(child.get("href")).groups()
            span = elements[position[first] : position[last or first] + 1]
            words += [e for e in span if e.tag == "w" and e.get("punc") != "true"]
        if not words:
            continue
        timed = [w for w in words if w.get("starttime")]
        result.append(
            {
                "session_id": uri,
                "speaker": speaker,
                "start_time": float(timed[0].get("starttime") if timed else segment.get("transcriber_start")),
                "end_time": float(timed[-1].get("endtime") if timed else segment.get("transcriber_end")),
                "words": " ".join(w.text for w in words),
            }
        )
    return result


def build(uri: str, data_dir: Path) -> list[dict]:
    result = []
    for agent, speaker in speakers(uri, data_dir).items():
        result += utterances(uri, agent, speaker, data_dir)
    return sorted(result, key=lambda u: (u["start_time"], u["speaker"]))


def load(uri: str, data_dir: Path = ami.DATA_DIR) -> list[dict]:
    return json.loads((references_dir(data_dir) / f"{uri}.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=ami.SPLITS, default=["test"])
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    args = parser.parse_args()

    out_dir = references_dir(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for uri in ami.meetings(split, data_dir=args.data_dir):
            transcript = build(uri, args.data_dir)
            (out_dir / f"{uri}.json").write_text(json.dumps(transcript, indent=1))
            n_words = sum(len(u["words"].split()) for u in transcript)
            print(f"{uri}: {len(transcript)} utterances, {n_words} words")


if __name__ == "__main__":
    main()
