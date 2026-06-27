# Breathing-AI-Model
# BreathingAI

AI model for detecting breathing type from WAV audio.

Supported classes:

- Apnea
- Hypopnea
- Normal

----------------------------------------

## Requirements

Python 3.10+

Install dependencies:

pip install -r requirements.txt

----------------------------------------

## Files

apnea_classifier.keras

    Trained CNN model.

predict.py

    Performs prediction from a WAV file.

requirements.txt

    Required Python packages.

----------------------------------------

## How to Predict

Open terminal.

Run:

python predict.py path_to_audio.wav

Example:

python predict.py test_audio/normal_00000.wav

----------------------------------------

Example Output

{
    "success": true,
    "class": "Normal",
    "confidence": 73.25,
    "probabilities":
    {
        "Apnea":12.10,
        "Hypopnea":14.65,
        "Normal":73.25
    }
}
