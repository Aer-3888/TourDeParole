# TourDeParole

Speaker-attributed meeting transcription ("who said what") evaluated on the AMI Meeting Corpus.

Three diarization systems are compared on the AMI test set (16 meetings, 9.06 h, Mix-Headset audio): the pretrained pyannote pipeline, a clustering baseline built from separate parts (Silero VAD, ECAPA-TDNN speaker embeddings, agglomerative clustering), and the same baseline with overlap handling. Every parameter is chosen on the dev set, and the test set is scored once.

| System | DER |
|--------|-----|
| pyannote community-1 | 17.05% |
| Clustering baseline | 30.42% |
| Clustering baseline with overlap handling | 26.13% |

DER without collar and with overlapped speech scored. Whisper large-v3 transcribes the meetings with 27.57% WER. The other scoring settings and the error analysis are in [Results](#results).

In progress: speaker-attributed transcripts scored with cpWER, and meeting notes written by a local LLM.

## Layout

Each module of `src/tourdeparole/` is a command, `python -m tourdeparole.<module>`. Model outputs go to `outputs/` and scores to `results/`.

| Module | Role |
|--------|------|
| `download_ami` | AMI diarization setup, manual annotations and audio |
| `references` | Speaker-attributed reference transcripts |
| `transcribe` | Whisper large-v3 transcription with word timestamps |
| `diarize_pyannote` | Pretrained pyannote diarization |
| `vad` | Speech detection with Silero VAD |
| `embed` | ECAPA-TDNN embeddings on sliding windows |
| `cluster` | Agglomerative clustering of the embeddings into speakers |
| `overlap` | Overlapped speech detection and second speaker assignment |
| `der`, `wer` | Scoring |
| `analysis` | Overlapped speech, speaker counts and errors by turn length |

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

### Speaker embeddings

Each speech region is cut into 1.5 s windows every 0.75 s, the last window ending with the region. Regions shorter than 1.5 s form a single window, and regions shorter than 0.25 s get none and later take the speaker of the nearest window. Each window is mapped to a 192-dimensional embedding by the ECAPA-TDNN model [speechbrain/spkrec-ecapa-voxceleb](https://hf.co/speechbrain/spkrec-ecapa-voxceleb), trained on VoxCeleb for speaker verification: windows from the same speaker should have a high cosine similarity.

```
python -m tourdeparole.embed --split test --vad outputs/vad-silero
```

Embeddings, window boundaries and speech regions are written per meeting to `outputs/ecapa/<vad>/<meeting>.npz`. The 18.7 h of dev and test audio take about 3 min per VAD on the GPU.

### Clustering

Agglomerative hierarchical clustering (AHC) of the windows of each meeting with average linkage on cosine distances: every window starts as its own cluster, and the two closest clusters are merged until the closest pair is further apart than a threshold, which sets the number of speakers. The embeddings are centered on their meeting mean first. Without centering they share a large common component, and on dev with the reference number of speakers most meetings end up as one large cluster plus a few outlier windows (28.6% confusion against 7.1% with centering, oracle speech).

Each window then gives its speaker to the part of its speech region closest to its center, so speaker changes fall halfway between two window centers.

The threshold is the only parameter tuned for clustering. It was swept from 0.80 to 1.20 by steps of 0.02 on the dev set, scoring DER in the full setting (`results/ahc_tuning_<vad>_dev.csv`). Both speech detections give the lowest DER at 0.98, with the number of speakers right in 10 of 18 dev meetings with oracle speech and 8 of 18 with Silero. Only confusion depends on the threshold, and it rises faster above the best value than below it: with Silero, DER is 31.41% at 0.90, where speakers are split into extra clusters, and 44.06% at 1.06, where different speakers start to be merged.

```
python -m tourdeparole.cluster --split dev --embeddings outputs/ecapa/vad-silero --tune
python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-silero
python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-silero --known-speakers
```

RTTMs are written to `outputs/ecapa-ahc-<vad>/` and scored with `tourdeparole.der`. With `--known-speakers` the tree is cut at the reference number of speakers instead of the threshold, which separates speaker counting errors from embedding errors.

## Overlap handling

The clustering baseline misses the second speaker of every overlap. This step detects overlapped speech and adds a second speaker there, following the approach of Bullock et al. (ICASSP 2020).

### Overlapped speech detection

The segmentation model of pyannote community-1 classifies each frame of a 10 s chunk into no speaker, one of three local speakers, or a pair of them. The probability of overlap is the sum of the three pair classes, averaged over chunks sliding by 1 s, and frames above a threshold are marked as overlapped. Only this model is reused from pyannote, not its clustering.

```
python -m tourdeparole.overlap detect --split test
```

Scores are cached per meeting in `outputs/overlap-scores/`, and precision and recall against the reference overlapped speech go to `results/overlap_detection_<split>.csv`.

### Second speaker

Inside each detected overlap, every turn of the baseline gets a second speaker. Two rules were compared on dev:

- temporal: the speaker of the closest turn of another speaker, before or after, usually the person who just spoke or is about to
- embedding: the other cluster whose centroid is closest to the mean embedding of the windows over the overlap

The rule and the detection threshold were chosen together on dev, by DER in the full setting (`results/overlap_tuning_<system>_dev.csv`). The temporal rule is better at every threshold, by 0.5 to 1.4 points, and the best threshold is 0.6 with both speech detections. DER changes by less than 0.1 point between 0.5 and 0.7. The same sweep with the reference overlapped speech gives the best result this step can reach with perfect overlap detection.

| Baseline speech detection | Overlap | DER on dev |
|---------------------------|---------|------------|
| Silero | none | 28.05% |
| Silero | detected, threshold 0.6 | 23.80% |
| Silero | reference | 20.26% |
| oracle | none | 20.20% |
| oracle | detected, threshold 0.6 | 16.00% |
| oracle | reference | 12.56% |

```
python -m tourdeparole.overlap tune --split dev --hyp outputs/ecapa-ahc-silero --embeddings outputs/ecapa/vad-silero
python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-silero
python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-silero --reference
```

RTTMs are written to `outputs/<system>-overlap/`, or `outputs/<system>-reference-overlap/` with the reference overlapped speech.

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

Full setting (no collar, overlapped speech scored):

| System | Speech detection | DER | Missed | False alarm | Confusion |
|--------|------------------|-----|--------|-------------|-----------|
| pyannote community-1 | pyannote | 17.05% | 9.52% | 3.58% | 3.95% |
| ECAPA-TDNN + AHC | Silero | 30.42% | 19.43% | 3.64% | 7.36% |
| ECAPA-TDNN + AHC | oracle | 23.50% | 14.55% | 0.00% | 8.95% |
| ECAPA-TDNN + AHC + overlap handling | Silero | 26.13% | 12.34% | 4.48% | 9.32% |
| ECAPA-TDNN + AHC + overlap handling | oracle | 19.62% | 7.45% | 0.83% | 11.34% |
| ECAPA-TDNN + AHC + reference overlap | Silero | 22.22% | 7.14% | 3.64% | 11.45% |
| ECAPA-TDNN + AHC + reference overlap | oracle | 15.87% | 2.09% | 0.00% | 13.78% |
| ECAPA-TDNN + AHC, known number of speakers | Silero | 30.94% | 19.43% | 3.64% | 7.87% |
| ECAPA-TDNN + AHC, known number of speakers | oracle | 22.97% | 14.55% | 0.00% | 8.42% |

DER in the three settings:

| System | Speech detection | Full | Fair | Forgiving |
|--------|------------------|------|------|-----------|
| pyannote community-1 | pyannote | 17.05% | 10.76% | 6.43% |
| ECAPA-TDNN + AHC | Silero | 30.42% | 22.04% | 13.93% |
| ECAPA-TDNN + AHC | oracle | 23.50% | 17.44% | 8.15% |
| ECAPA-TDNN + AHC + overlap handling | Silero | 26.13% | 18.82% | 14.30% |
| ECAPA-TDNN + AHC + overlap handling | oracle | 19.62% | 14.52% | 8.53% |

The pyannote DER matches the 17.0% reported on the model card for AMI Mix-Headset, which validates the data and scoring setup. Missed speech is its largest error, and 71% of it (2075 of 2925 s) falls in overlapped regions, where the pipeline finds fewer speakers than are talking. The predicted number of speakers is correct in 12 of 16 meetings. In the other 4, one extra speaker is predicted, with 10 to 76 s of speech.

The clustering baseline has 13.4 points more DER than pyannote, and missed speech is most of the gap. The baseline gives one speaker to each moment, so the second speaker of every overlap is lost: 4520 of its 5969 s of missed speech are in overlapped regions, against 2075 s for pyannote, which detects part of the overlapped speech. With oracle speech detection, the missed speech is exactly the overlapped speech that a single speaker cannot cover (14.55%), and only confusion is left to compare: 8.95% against 3.95% for pyannote. Silero adds 1449 s of missed speech outside overlaps and about as much false alarm as pyannote.

Giving the reference number of speakers does not help (30.94% against 30.42% with Silero), although the threshold finds the right number in only 9 of 16 meetings. The confusion comes from the embeddings rather than from counting speakers.

Overlap handling lowers the DER of the baseline by 4.3 points with Silero (26.13%) and 3.9 points with oracle speech (19.62%), as on dev (4.3 and 4.2). Missed speech in overlapped regions falls from 4520 to 2341 s. The second speaker is not always the right one, so confusion rises from 7.36% to 9.32%, and with the reference overlapped speech confusion becomes the largest error left (11.45%). The detector is conservative on test, with 89.4% precision and 57.0% recall at the threshold chosen on dev, and perfect overlap detection would bring the system to 22.22%. In the forgiving setting, where overlapped speech is not scored, DER rises slightly (13.93% to 14.30%) because of second speakers added in false overlap detections. The gap with pyannote goes from 13.4 to 9.1 points.

Share of reference turn time given to the right speaker, by turn length, with the share of that time where another speaker talks too:

| Turn length | Turns | Overlapped | pyannote | ECAPA-TDNN + AHC, Silero | ECAPA-TDNN + AHC, oracle | + overlap handling, Silero |
|-------------|-------|------------|----------|--------------------------|--------------------------|----------------------------|
| < 1 s  | 3247 | 69.6% | 40.7% | 18.5% | 22.6% | 32.1% |
| 1-2 s  | 1080 | 56.8% | 59.5% | 46.5% | 50.1% | 56.9% |
| 2-5 s  | 1376 | 42.6% | 76.6% | 63.5% | 66.1% | 71.5% |
| > 5 s  | 1790 | 19.4% | 93.1% | 80.2% | 83.6% | 84.0% |

Short turns are mostly overlapped speech: 70% of the time in turns under 1 s overlaps another speaker, typically a backchannel such as "yeah" over the current speaker. Both systems lose most of these turns to the speaker they overlap, and the baseline more so. A 1.5 s window is longer than half of the reference turns (median 1.38 s), so the embedding of a short turn mostly describes the speech around it. On turns over 5 s, the baseline still gives 16.4% of the time to another speaker with oracle speech, against 4.3% for pyannote. Part of it is the 19.4% of overlapped time in these turns, where a single speaker per moment can only be right for one of the two. Overlap handling helps the short turns most: the share of time given to the right speaker in turns under 1 s goes from 18.5% to 32.1%.

### Transcription

| System | WER | Substitutions | Deletions | Insertions |
|--------|-----|---------------|-----------|------------|
| Whisper large-v3 | 27.57% | 5.65% | 18.88% | 3.05% |

Deletions account for two thirds of the errors. Across meetings, WER follows the share of overlapped speech closely (Pearson r = 0.97 over 16 meetings): the four EN2002 meetings have 24 to 30% overlapped speech and 34 to 41% WER, the others have 4 to 16% and 16 to 31%. The EN2002 meetings are also the only unscripted ones in the test set, so meeting type and overlap are confounded here.

### Runtime

On an RTX 4060 Laptop GPU (8 GB): Whisper large-v3 took 100 min for the 9.06 h of audio, pyannote community-1 took 19 min. The clustering baseline took under 2 min: 55 s for Silero on the CPU, 58 s for the embeddings and 3 s for the clustering, model loading excluded. Overlap detection took 2 min for the test set, model loading included.

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

Clustering baseline, tuned on dev and scored on test:

```
python -m tourdeparole.vad --split dev --tune
python -m tourdeparole.vad --split dev --method silero
python -m tourdeparole.vad --split dev --method oracle
python -m tourdeparole.vad --split test --method silero
python -m tourdeparole.vad --split test --method oracle
python -m tourdeparole.embed --split dev --vad outputs/vad-silero
python -m tourdeparole.embed --split dev --vad outputs/vad-oracle
python -m tourdeparole.embed --split test --vad outputs/vad-silero
python -m tourdeparole.embed --split test --vad outputs/vad-oracle
python -m tourdeparole.cluster --split dev --embeddings outputs/ecapa/vad-silero --tune
python -m tourdeparole.cluster --split dev --embeddings outputs/ecapa/vad-oracle --tune
python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-silero
python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-oracle
python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-silero --known-speakers
python -m tourdeparole.cluster --split test --embeddings outputs/ecapa/vad-oracle --known-speakers
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-silero
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-oracle
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-silero-known-speakers
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-oracle-known-speakers
python -m tourdeparole.analysis diarization --split test --hyp outputs/ecapa-ahc-silero
python -m tourdeparole.analysis diarization --split test --hyp outputs/ecapa-ahc-oracle
```

Overlap handling, on top of the clustering baseline:

```
python -m tourdeparole.overlap detect --split dev
python -m tourdeparole.overlap detect --split test
python -m tourdeparole.cluster --split dev --embeddings outputs/ecapa/vad-silero
python -m tourdeparole.cluster --split dev --embeddings outputs/ecapa/vad-oracle
python -m tourdeparole.overlap tune --split dev --hyp outputs/ecapa-ahc-silero --embeddings outputs/ecapa/vad-silero
python -m tourdeparole.overlap tune --split dev --hyp outputs/ecapa-ahc-oracle --embeddings outputs/ecapa/vad-oracle
python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-silero
python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-oracle
python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-silero --reference
python -m tourdeparole.overlap assign --split test --hyp outputs/ecapa-ahc-oracle --reference
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-silero-overlap
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-oracle-overlap
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-silero-reference-overlap
python -m tourdeparole.der --split test --hyp outputs/ecapa-ahc-oracle-reference-overlap
python -m tourdeparole.analysis diarization --split test --hyp outputs/ecapa-ahc-silero-overlap
python -m tourdeparole.analysis diarization --split test --hyp outputs/ecapa-ahc-oracle-overlap
```

## License

The code is released under the MIT license. The AMI Meeting Corpus is distributed under CC BY 4.0 and is not part of this repository.
