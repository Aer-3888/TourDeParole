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

## Transcription

Whisper large-v3 through faster-whisper (float16 on GPU), English, beam size 5, word timestamps and the built-in Silero VAD filter, on the full meeting audio:

```
python -m tourdeparole.transcribe --split test
```

Each meeting is written to `outputs/whisper-large-v3/<meeting>.json` with segments, words and timestamps.

## Scoring

### DER

Diarization error rate is computed with pyannote.metrics over the full UEM of each meeting and accumulated over the split (total error time divided by total reference speech time). The three settings follow the naming of Landini et al. (VBx, 2022), who released this AMI setup:

| Setting   | Collar          | Overlapped speech |
|-----------|-----------------|-------------------|
| full      | none            | scored            |
| fair      | 0.25 s per side | scored            |
| forgiving | 0.25 s per side | not scored        |

The full setting is the one used in the pyannote benchmarks and is the main number in this project. pyannote.metrics takes the total collar width, so 0.25 s per side is `collar=0.5`.

```
python -m tourdeparole.der --split test --hyp outputs/<system>
```

Results go to `results/der_<system>_<split>.csv` with missed detection, false alarm and confusion per meeting.

### WER

Word error rate ignores speakers. The reference is every utterance of the meeting concatenated in order of start time, the hypothesis is the Whisper output in order, and both go through the Whisper English text normalizer (lowercase, no punctuation, filler words such as "um" and "hmm" removed, standard spellings). Errors are counted with jiwer and accumulated over the split.

Concatenating speakers in time order penalizes overlapped speech, where the order of words from two speakers is ambiguous. The speaker-attributed metric cpWER comes later.

```
python -m tourdeparole.wer --split test --hyp outputs/whisper-large-v3
```
