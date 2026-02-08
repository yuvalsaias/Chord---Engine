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

app = FastAPI()

# ---------------------------
# Chord data
# ---------------------------
CHORDS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

CHORD_TEMPLATES = {
    "maj":[0,4,7],
    "min":[0,3,7],
    "dim":[0,3,6],
    "aug":[0,4,8],

    "7":[0,4,7,10],
    "maj7":[0,4,7,11],
    "m7":[0,3,7,10],
    "mMaj7":[0,3,7,11],

    "sus2":[0,2,7],
    "sus4":[0,5,7],

    "add9":[0,4,7,14],
    "madd9":[0,3,7,14],

    "9":[0,4,7,10,14],
    "m9":[0,3,7,10,14],

    "6":[0,4,7,9],
    "m6":[0,3,7,9]
}

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# ---------------------------
# Bass detection
# ---------------------------
def detect_bass_note(segment, sr):

    S = np.abs(librosa.stft(segment))
    freqs = librosa.fft_frequencies(sr=sr)

    bass_energy = np.mean(S[freqs < 200], axis=1)

    if len(bass_energy) == 0:
        return None

    idx = np.argmax(bass_energy)
    freq = freqs[idx]

    midi = librosa.hz_to_midi(freq)
    note = int(round(midi)) % 12

    return NOTE_NAMES[note]

# ---------------------------
# Chord detection
# ---------------------------
def detect_chord_from_chroma(chroma):

    best_score = 0
    best_chord = "N"

    if np.sum(chroma) == 0:
        return "N"

    chroma = chroma / np.sum(chroma)

    for root in range(12):

        for quality, intervals in CHORD_TEMPLATES.items():

            template = np.zeros(12)

            for interval in intervals:
                template[(root + interval) % 12] = 1

            template = template / np.sum(template)

            score = np.dot(chroma, template)

            if score > best_score:
                best_score = score
                best_chord = NOTE_NAMES[root] + quality

    return best_chord

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

        chord = detect_chord_from_chroma(chroma_mean)

        bass = detect_bass_note(segment, sr)
        if bass and not chord.startswith(bass):
            chord = chord + "/" + bass

        chords.append({
            "beat_time": float(beat_times[i]),
            "chord": chord
        })

    return chords

# ---------------------------
# Basic smoothing
# ---------------------------
def smooth_chords(chords):

    smoothed = []

    for i in range(len(chords)):

        curr = chords[i]["chord"]

        if i > 0 and i < len(chords)-1:

            prev = chords[i-1]["chord"]
            next = chords[i+1]["chord"]

            if prev == next and curr != prev:
                curr = prev

        smoothed.append({
            "beat_time": chords[i]["beat_time"],
            "chord": curr
        })

    return smoothed

# ---------------------------
# Harmonic stability filter
# ---------------------------
def harmonic_filter(chords, min_beats=2):

    if not chords:
        return chords

    filtered = []

    current = chords[0]
    count = 1

    for i in range(1, len(chords)):

        if chords[i]["chord"] == current["chord"]:
            count += 1
        else:

            if count >= min_beats:
                filtered.extend([current] * count)
            else:
                if filtered:
                    filtered.extend([filtered[-1]] * count)
                else:
                    filtered.extend([current] * count)

            current = chords[i]
            count = 1

    filtered.extend([current] * count)

    return filtered

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

        y, sr = librosa.load(audio_path, sr=None)

        beat_times = detect_beats(audio_path)

        chords = chords_per_beat(y, sr, beat_times)
        chords = smooth_chords(chords)
        chords = harmonic_filter(chords)

        bars = group_bars(chords)

        return {"bars": bars}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
