#!/usr/bin/env bash
# Runs the whole pipeline in order. Parameters are tuned on dev and test is scored once.
set -euo pipefail

tdp() {
    local module=$1
    shift
    python -m "tourdeparole.$module" "$@"
}

# data, transcription and pyannote
tdp download_ami --splits test dev
tdp references --splits test dev
tdp transcribe --split test
tdp diarize_pyannote --split test
tdp wer --split test --hyp outputs/whisper-large-v3
tdp der --split test --hyp outputs/pyannote-community-1
tdp analysis data --split test
tdp analysis diarization --split test --hyp outputs/pyannote-community-1
tdp analysis transcription --split test --wer results/wer_whisper-large-v3_test.csv

# clustering baseline
tdp vad --split dev --tune
for split in dev test; do
    for vad in silero oracle; do
        tdp vad --split "$split" --method "$vad"
        tdp embed --split "$split" --vad "outputs/vad-$vad"
    done
done
for vad in silero oracle; do
    tdp cluster --split dev --embeddings "outputs/ecapa/vad-$vad" --tune
    tdp cluster --split dev --embeddings "outputs/ecapa/vad-$vad"
    tdp cluster --split test --embeddings "outputs/ecapa/vad-$vad"
    tdp cluster --split test --embeddings "outputs/ecapa/vad-$vad" --known-speakers
    tdp der --split test --hyp "outputs/ecapa-ahc-$vad"
    tdp der --split test --hyp "outputs/ecapa-ahc-$vad-known-speakers"
    tdp analysis diarization --split test --hyp "outputs/ecapa-ahc-$vad"
done

# overlap handling
tdp overlap detect --split dev
tdp overlap detect --split test
for vad in silero oracle; do
    tdp overlap tune --split dev --hyp "outputs/ecapa-ahc-$vad" --embeddings "outputs/ecapa/vad-$vad"
    tdp overlap assign --split test --hyp "outputs/ecapa-ahc-$vad"
    tdp overlap assign --split test --hyp "outputs/ecapa-ahc-$vad" --reference
    tdp der --split test --hyp "outputs/ecapa-ahc-$vad-overlap"
    tdp der --split test --hyp "outputs/ecapa-ahc-$vad-reference-overlap"
    tdp analysis diarization --split test --hyp "outputs/ecapa-ahc-$vad-overlap"
done

# README figures
tdp plot der
tdp plot timeline --meeting EN2002d --start 2055 --duration 60
