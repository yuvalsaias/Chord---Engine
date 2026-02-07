from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from chordmini.inference import analyze_audio_file

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    result = analyze_audio_file(temp_path)

    os.remove(temp_path)

    return result
