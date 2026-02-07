from fastapi import FastAPI, UploadFile, File
import tempfile
import madmom
import librosa
import numpy as np

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    
    # שמירת קובץ זמני
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # טעינת האודיו
    y, sr = librosa.load(temp_path)

    # דוגמה בסיסית לזיהוי אקורדים (placeholder)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chords = np.mean(chroma, axis=1)

    return {
        "chords_vector": chords.tolist()
    }
