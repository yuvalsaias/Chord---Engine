import os
import tempfile
from fastapi import FastAPI, UploadFile, File

from omnizart.chord import app as chord_app
from omnizart.utils import io

app = FastAPI()

# ---------------------------
# Convert Omnizart output to chart bars
# ---------------------------

def omnizart_to_chart(chord_list):

    bars = []
    current_bar = []
    beat_counter = 0
    bar_number = 1

    for chord in chord_list:

        label = chord[2]

        if label == "N":
            continue

        current_bar.append({
            "beat": beat_counter + 1,
            "chord": label
        })

        beat_counter += 1

        if beat_counter == 4:

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

        # 🔥 Deep ML chord recognition
        result = chord_app.transcribe(audio_path)

        # result format:
        # [ (start_time, end_time, chord_label), ... ]

        bars = omnizart_to_chart(result)

        return {
            "bars": bars,
            "raw_chords": result
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
