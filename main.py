import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import numpy as np
import librosa
import madmom.features.beats as beats
import crema

app = FastAPI()

# ---------------------------------------------------
# Beat detection (Madmom)
# ---------------------------------------------------
def detect_beats(audio_path):

    proc = beats.RNNBeatProcessor()
    act = proc(audio_path)

    tracker = beats.DBNBeatTrackingProcessor(fps=100)
    beat_times = tracker(act)

    return beat_times


# ---------------------------------------------------
# CREMA chord detection (FIXED)
# ---------------------------------------------------
def detect_chords_crema(audio_path):

    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    model = crema.models.chord.ChordModel()
    result = model.predict(y, sr)

    chords = []

    for interval, label in zip(result.intervals, result.labels):

        chords.append({
            "start": float(interval[0]),
            "end": float(interval[1]),
            "chord": label
        })

    return chords


# ---------------------------------------------------
# Align chords to beats
# ---------------------------------------------------
def align_chords_to_beats(chord_segments, beat_times):

    beat_chords = []

    for beat in beat_times:

        current = "N"

        for seg in chord_segments:
            if seg["start"] <= beat < seg["end"]:
                current = seg["chord"]
                break

        beat_chords.append({
            "beat_time": float(beat),
            "chord": current
        })

    return beat_chords


# ---------------------------------------------------
# Remove duplicate consecutive chords
# ---------------------------------------------------
def compress_chords(chords):

    compressed = []

    last = None

    for c in chords:

        if c["chord"] != last:
            compressed.append(c)

        last = c["chord"]

    return compressed


# ---------------------------------------------------
# Group into bars
# ---------------------------------------------------
def group_bars(chords, beats_per_bar=4):

    bars = []
    current_bar = []
    beat_counter = 0
    bar_number = 1

    for c in chords:

        current_bar.append({
            "beat": beat_counter + 1,
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


# ---------------------------------------------------
# API Endpoint
# ---------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    tmp = tempfile.NamedTemporaryFile(delete=False)
    audio_path = tmp.name

    try:

        content = await file.read()
        tmp.write(content)
        tmp.close()

        # --- CREMA harmonic detection ---
        chord_segments = detect_chords_crema(audio_path)

        # --- Beat detection ---
        beat_times = detect_beats(audio_path)

        # --- Align ---
        beat_chords = align_chords_to_beats(chord_segments, beat_times)

        # --- Remove jitter ---
        beat_chords = compress_chords(beat_chords)

        # --- Bars ---
        bars = group_bars(beat_chords)

        return {
            "bars": bars,
            "segments": chord_segments
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
