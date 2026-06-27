import os
import sys
import json
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------
# Settings
# -----------------------------

MODEL_PATH = "apnea_classifier.keras"

N_MFCC = 40
MAX_LEN = 400

CLASS_NAMES = [
    "Apnea",
    "Hypopnea",
    "Normal"
]

# -----------------------------
# Check input
# -----------------------------

if len(sys.argv) != 2:

    print(json.dumps({
        "success": False,
        "error": "Usage: python predict.py audio.wav"
    }))

    sys.exit()

audio_path = sys.argv[1]

if not os.path.exists(audio_path):

    print(json.dumps({
        "success": False,
        "error": "Audio file not found"
    }))

    sys.exit()

# -----------------------------
# Load model
# -----------------------------

model = load_model(MODEL_PATH)

# -----------------------------
# Read audio
# -----------------------------

signal, sr = librosa.load(
    audio_path,
    sr=16000
)

# -----------------------------
# MFCC
# -----------------------------

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=N_MFCC
)

# -----------------------------
# Pad / Crop
# -----------------------------

if mfcc.shape[1] < MAX_LEN:

    pad = MAX_LEN - mfcc.shape[1]

    mfcc = np.pad(
        mfcc,
        ((0,0),(0,pad)),
        mode="constant"
    )

else:

    mfcc = mfcc[:, :MAX_LEN]

# -----------------------------
# CNN Input Shape
# -----------------------------

mfcc = np.expand_dims(mfcc, axis=0)
mfcc = np.expand_dims(mfcc, axis=-1)

# -----------------------------
# Predict
# -----------------------------

prediction = model.predict(mfcc, verbose=0)[0]

class_id = int(np.argmax(prediction))

confidence = float(prediction[class_id] * 100)

# -----------------------------
# Output JSON
# -----------------------------

result = {

    "success": True,

    "class": CLASS_NAMES[class_id],

    "confidence": round(confidence, 2),

    "probabilities": {

        "Apnea": round(float(prediction[0] * 100), 2),

        "Hypopnea": round(float(prediction[1] * 100), 2),

        "Normal": round(float(prediction[2] * 100), 2)

    }

}

print(json.dumps(result))
