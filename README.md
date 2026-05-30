# Task-2-AI-Speech-Recognition-Studio

A basic AI Speech-to-Text Studio that transcribes short WAV audio clips using pre-trained models.

## Files

- `app.py`: Command-line application for transcription.
- `requirements.txt`: Python dependencies.

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Transcribe an audio file using the default Wav2Vec2 backend:

```bash
python app.py path/to/audio.wav
```

To use the `SpeechRecognition` Google-based backend instead:

```bash
python app.py path/to/audio.wav --method speech_recognition
```

> Note: This app currently supports WAV files. For best results, use a short audio clip sampled at 16 kHz.
