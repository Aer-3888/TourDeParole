# TourDeParole

Speaker-attributed meeting transcription ("who said what") evaluated on the AMI Meeting Corpus.

## Setup

Requires Python 3.12 and an NVIDIA GPU with CUDA.

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Data

The evaluation follows the standard AMI diarization setup from [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup) (pinned commit `67c2d53`): official Full-corpus-ASR split, `only_words` reference RTTMs and Mix-Headset audio.

```
python -m tourdeparole.download_ami --splits test dev
```

This writes the setup, the AMI manual annotations 1.6.2 and the audio to `data/` (about 2.2 GB for test and dev).

| Split | Meetings | Duration | Speakers per meeting | Overlapped speech |
|-------|----------|----------|----------------------|-------------------|
| dev   | 18       | 9.67 h   | 4                    | 14.1% |
| test  | 16       | 9.06 h   | 3 or 4               | 14.6% |

Overlapped speech is the share of speech time where at least two people talk at once, computed from the reference RTTMs.

Reference transcripts come from the word-level manual annotations, grouped into the official AMI segments and attributed to the same speaker names as the RTTMs:

```
python -m tourdeparole.references --splits test dev
```

Each meeting is written to `data/references/<meeting>.json` in SegLST format (the JSON format used by meeteval). Punctuation tokens are dropped, and disfluencies and truncated words are kept as transcribed.
