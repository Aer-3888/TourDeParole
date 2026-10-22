"""Download the AMI diarization setup, manual annotations and Mix-Headset audio.

    python -m tourdeparole.download_ami --splits test dev
"""

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tourdeparole import ami

SETUP_COMMIT = "67c2d539286e89f68952d5dcf83912bd9f01dfae"
SETUP_URL = f"https://github.com/pyannote/AMI-diarization-setup/archive/{SETUP_COMMIT}.zip"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{uri}/audio/{uri}.Mix-Headset.wav"
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def download_zip(url: str) -> zipfile.ZipFile:
    with urllib.request.urlopen(url) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def download_setup(data_dir: Path) -> None:
    target = ami.setup_dir(data_dir)
    if target.exists():
        return
    print(f"Downloading AMI-diarization-setup at {SETUP_COMMIT[:7]}")
    data_dir.mkdir(parents=True, exist_ok=True)
    download_zip(SETUP_URL).extractall(data_dir)
    (data_dir / f"AMI-diarization-setup-{SETUP_COMMIT}").rename(target)


def download_annotations(data_dir: Path) -> None:
    target = ami.annotations_dir(data_dir)
    if target.exists():
        return
    print("Downloading AMI manual annotations 1.6.2")
    download_zip(ANNOTATIONS_URL).extractall(target)


def download_audio(uri: str, data_dir: Path) -> None:
    target = ami.audio_path(uri, data_dir)
    if target.exists():
        return
    print(f"Downloading {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(AUDIO_URL.format(uri=uri)) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=ami.SPLITS, default=["test"])
    parser.add_argument("--mini", action="store_true", help="only the 3-meeting mini lists")
    parser.add_argument("--data-dir", type=Path, default=ami.DATA_DIR)
    args = parser.parse_args()

    download_setup(args.data_dir)
    download_annotations(args.data_dir)
    for split in args.splits:
        for uri in ami.meetings(split, args.mini, args.data_dir):
            download_audio(uri, args.data_dir)


if __name__ == "__main__":
    main()
