import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence

import numpy as np
np.float = float
np.int = int

from fastapi import FastAPI, UploadFile, File
import tempfile
import librosa

app = FastAPI()

# --------------------------
# Simple chord templates
# --------------------------

CHORDS = {
    "C": [1,0,0,0,1,0,0,1,0,0,0,0],
    "G": [0,0,1,0,0,0,0,1,0,0,0,1],
    "Am": [0,0,0,1,0,0,0,0,1,0,0,0],
    "F": [1,0,0,0,0,1,0,0,1,0,0,0],
}

def detect_chord(chroma_vector):
    best = None
    best_score = -1

    for chord, template in CHORDS.items():
        score = np.dot(chroma_vector, template)
        if score > best_score:
            best_score = score
            best = chord

    return best


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    y, sr = librosa.load(temp_path)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    avg_chroma = np.mean(chroma, axis=1)

    chord = detect_chord(avg_chroma)

    return {
        "detected_chord": chord,
        "chroma_vector": avg_chroma.tolist()
    }
