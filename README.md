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

Overlapped speech is the share of speech time where at least two people talk at once, computed from the reference RTTMs:

```
python -m tourdeparole.analysis data --split test
```

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

## Diarization with pyannote

The pretrained [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) pipeline (pyannote.audio 4.0.7, default hyperparameters, number of speakers not given). It needs a Hugging Face token with the model conditions accepted (`hf auth login`).

```
python -m tourdeparole.diarize_pyannote --split test
```

RTTMs are written to `outputs/pyannote-community-1/`. The `exclusive/` subfolder holds the variant without overlapping turns, which pyannote provides for alignment with transcripts. pyannote usage telemetry is disabled in the script.

## Clustering baseline

A diarization system built from separate parts: speech detection, speaker embeddings on short windows, then clustering of the windows into speakers. It assigns a single speaker to each moment, so it cannot handle overlapped speech by design.

### Speech detection

[Silero VAD](https://github.com/snakers4/silero-vad) as bundled in faster-whisper, with the Silero defaults for minimum speech (250 ms) and minimum silence (100 ms). The speech threshold and the padding added around each region were chosen on the dev set by minimizing missed plus false alarm speech over a grid of 25 settings (`results/vad_tuning_dev.csv`): threshold 0.3 and 200 ms of padding.

```
python -m tourdeparole.vad --split dev --tune
python -m tourdeparole.vad --split test --method silero
python -m tourdeparole.vad --split test --method oracle
```

Speech regions are written as RTTMs to `outputs/vad-silero/`, and scores to `results/vad_silero_<split>.csv`. The oracle method takes the speech regions from the reference instead, which isolates the errors of the embeddings and the clustering.

| Split | Missed speech | False alarm |
|-------|---------------|-------------|
| dev   | 3.83% | 5.93% |
| test  | 5.71% | 4.26% |

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

### Error analysis

The diarization errors are split between overlapped speech and the rest of each meeting, and the predicted number of speakers is compared with the reference. Each reference turn is also checked against the hypothesis after the optimal speaker mapping, to see how much of it goes to the right speaker depending on its length.

```
python -m tourdeparole.analysis diarization --split test --hyp outputs/<system>
python -m tourdeparole.analysis transcription --split test --wer results/wer_<system>_test.csv
```

Results go to `results/regions_<system>_<split>.csv` (error seconds per meeting) and `results/turns_<system>_<split>.csv`.

## Results

AMI test set, 16 meetings, 9.06 h, Mix-Headset audio. Per-meeting numbers are in `results/`.

### Diarization

| System | Setting | DER | Missed | False alarm | Confusion |
|--------|---------|-----|--------|-------------|-----------|
| pyannote community-1 | full      | 17.05% | 9.52% | 3.58% | 3.95% |
| pyannote community-1 | fair      | 10.76% | 6.78% | 1.76% | 2.22% |
| pyannote community-1 | forgiving |  6.43% | 3.07% | 2.14% | 1.22% |

The full DER matches the 17.0% reported on the model card for AMI Mix-Headset, which validates the data and scoring setup. Missed speech is the largest error, and 71% of it (2075 of 2925 s) falls in overlapped regions, where the pipeline finds fewer speakers than are talking. The predicted number of speakers is correct in 12 of 16 meetings. In the other 4, one extra speaker is predicted, with 10 to 76 s of speech.

### Transcription

| System | WER | Substitutions | Deletions | Insertions |
|--------|-----|---------------|-----------|------------|
| Whisper large-v3 | 27.57% | 5.65% | 18.88% | 3.05% |

Deletions account for two thirds of the errors. Across meetings, WER follows the share of overlapped speech closely (Pearson r = 0.97 over 16 meetings): the four EN2002 meetings have 24 to 30% overlapped speech and 34 to 41% WER, the others have 4 to 16% and 16 to 31%. The EN2002 meetings are also the only unscripted ones in the test set, so meeting type and overlap are confounded here.

### Runtime

On an RTX 4060 Laptop GPU (8 GB): Whisper large-v3 took 100 min for the 9.06 h of audio, pyannote community-1 took 19 min.

## Reproduction

```
python -m tourdeparole.download_ami --splits test dev
python -m tourdeparole.references --splits test dev
python -m tourdeparole.transcribe --split test
python -m tourdeparole.diarize_pyannote --split test
python -m tourdeparole.wer --split test --hyp outputs/whisper-large-v3
python -m tourdeparole.der --split test --hyp outputs/pyannote-community-1
python -m tourdeparole.analysis data --split test
python -m tourdeparole.analysis diarization --split test --hyp outputs/pyannote-community-1
python -m tourdeparole.analysis transcription --split test --wer results/wer_whisper-large-v3_test.csv
```
