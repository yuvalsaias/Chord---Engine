import collections
import collections.abc

collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import tempfile
import madmom.features.beats as beats

app = FastAPI()

# ---------------------------
# Simple chord classifier
# ---------------------------
CHORDS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def classify_chord(chroma):
    return CHORDS[np.argmax(chroma)]

# ---------------------------
# Beat Detection
# ---------------------------
def detect_beats(audio_path):
    proc = beats.RNNBeatProcessor()
    act = proc(audio_path)

    tracker = beats.DBNBeatTrackingProcessor(fps=100)
    beat_times = tracker(act)

    return beat_times

# ---------------------------
# Chord detection per beat
# ---------------------------
def chords_per_beat(y, sr, beat_times):

    chords = []

    for i in range(len(beat_times)-1):

        start = int(beat_times[i] * sr)
        end = int(beat_times[i+1] * sr)

        segment = y[start:end]

        if len(segment) < sr * 0.2:
            continue

        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        chord = classify_chord(chroma_mean)

        chords.append({
            "beat_time": float(beat_times[i]),
            "chord": chord
        })

    return chords

# ---------------------------
# Group beats into bars
# ---------------------------
def group_bars(chords, beats_per_bar=4):

    bars = []
    current_bar = []
    beat_counter = 0
    bar_number = 1

    for c in chords:

        current_bar.append({
            "beat": beat_counter+1,
            "chord": c["chord"]
        })

        beat_counter += 1

        if beat_counter == beats_per_bar:

            bars.append({
                "bar": bar_number,
                "chords": current_bar
            })

            current_bar = []
            beat_counter = 0
            bar_number += 1

    return bars

# ---------------------------
# API
# ---------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    y, sr = librosa.load(audio_path)

    beat_times = detect_beats(audio_path)

    chords = chords_per_beat(y, sr, beat_times)

    bars = group_bars(chords)

    return {
        "bars": bars
    }
