"""AMI Meeting Corpus: meeting lists, audio paths and reference annotations."""

from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Timeline
from pyannote.database.util import load_rttm, load_uem

DATA_DIR = Path("data")
SPLITS = ("train", "dev", "test")
SAMPLE_RATE = 16000


def setup_dir(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "AMI-diarization-setup"


def annotations_dir(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "ami_manual"


def meetings(split: str, mini: bool = False, data_dir: Path = DATA_DIR) -> list[str]:
    name = f"{split}.mini.txt" if mini else f"{split}.meetings.txt"
    return (setup_dir(data_dir) / "lists" / name).read_text().split()


def audio_path(uri: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "amicorpus" / uri / "audio" / f"{uri}.Mix-Headset.wav"


def load_audio(uri: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    """Mono float32 waveform at 16 kHz, the native format of the Mix-Headset files."""
    audio, sample_rate = sf.read(audio_path(uri, data_dir), dtype="float32")
    assert sample_rate == SAMPLE_RATE and audio.ndim == 1
    return audio


def reference(uri: str, split: str, data_dir: Path = DATA_DIR) -> Annotation:
    path = setup_dir(data_dir) / "only_words" / "rttms" / split / f"{uri}.rttm"
    return load_rttm(path)[uri]


def uem(uri: str, split: str, data_dir: Path = DATA_DIR) -> Timeline:
    path = setup_dir(data_dir) / "uems" / split / f"{uri}.uem"
    return load_uem(path)[uri]
