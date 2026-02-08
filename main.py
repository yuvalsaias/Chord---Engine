import collections
import collections.abc
import numpy as np
import os

collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from fastapi import FastAPI, UploadFile, File
import librosa
import tempfile
import madmom.features.beats as beats
import crema

app = FastAPI()

# ---------------------------
# Beat Detection (נשאר)
# ---------------------------
def detect_beats(audio_path):

    proc = beats.RNNBeatProcessor()
    act = proc(audio_path)

    tracker = beats.DBNBeatTrackingProcessor(fps=100)
    beat_times = tracker(act)

    return beat_times

# ---------------------------
# CREMA Chord Detection
# ---------------------------
def detect_chords_crema(audio_path):

    result = crema.analyze(audio_path)

    chords = []

    for entry in result["chords"]:

        chords.append({
            "start": float(entry["start"]),
            "end": float(entry["end"]),
            "chord": entry["label"]
        })

    return chords

# ---------------------------
# Map chords onto beats
# ---------------------------
def map_chords_to_beats(chord_segments, beat_times):

    mapped = []

    for beat in beat_times:

        current_chord = "N"

        for seg in chord_segments:
            if seg["start"] <= beat < seg["end"]:
                current_chord = seg["chord"]
                break

        mapped.append({
            "beat_time": float(beat),
            "chord": current_chord
        })

    return mapped

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

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    tmp = tempfile.NamedTemporaryFile(delete=False)
    audio_path = tmp.name

    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # Beat detection
        beat_times = detect_beats(audio_path)

        # CREMA chords
        chord_segments = detect_chords_crema(audio_path)

        # Map chords onto beats
        chords = map_chords_to_beats(chord_segments, beat_times)

        # Bars
        bars = group_bars(chords)

        return {"bars": bars}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
