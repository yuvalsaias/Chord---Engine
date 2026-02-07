import collections
import collections.abc

# Fix for old madmom compatibility
collections.MutableSequence = collections.abc.MutableSequence

import numpy as np

# Fix numpy deprecations used by madmom
np.float = float
np.int = int

from fastapi import FastAPI, UploadFile, File
import tempfile
import madmom
import librosa

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # שמירת קובץ זמני
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # טעינת האודיו
    y, sr = librosa.load(temp_path)

    # דוגמה בסיסית (כרגע רק chroma)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chords_vector = np.mean(chroma, axis=1)

    return {
        "chords_vector": chords_vector.tolist()
    }
