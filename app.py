import argparse
import os
import sys


def transcribe_with_speech_recognition(audio_path: str) -> str:
    try:
        import speech_recognition as sr
    except ImportError:
        raise RuntimeError("SpeechRecognition is not installed. Install it with `pip install SpeechRecognition`.")

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data)
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech recognition service failed: {exc}")
    except sr.UnknownValueError:
        return "(Unable to recognize speech in the audio clip.)"


def transcribe_with_wav2vec(audio_path: str, model_name: str = "facebook/wav2vec2-base-960h") -> str:
    try:
        from transformers import pipeline
    except ImportError:
        raise RuntimeError("transformers is not installed. Install it with `pip install transformers torch scipy`.")

    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Install it with `pip install torch`.")

    from scipy.io import wavfile
    import numpy as np

    sample_rate, speech = wavfile.read(audio_path)
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)

    if speech.dtype != np.float32:
        if np.issubdtype(speech.dtype, np.integer):
            max_value = np.iinfo(speech.dtype).max
            speech = speech.astype(np.float32) / max_value
        else:
            speech = speech.astype(np.float32)

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=model_name, device=device)
    result = asr(speech, chunk_length_s=20)
    return result.get("text", "").strip()


def validate_audio_path(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Audio file does not exist: {path}")
    if not path.lower().endswith(".wav"):
        raise argparse.ArgumentTypeError("Only WAV files are supported. Please provide a .wav file.")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Speech-to-Text Studio: Transcribe a short audio clip using SpeechRecognition or Wav2Vec2."
    )
    parser.add_argument("audio", type=validate_audio_path, help="Path to the WAV audio file to transcribe.")
    parser.add_argument(
        "--method",
        choices=["speech_recognition", "wav2vec"],
        default="wav2vec",
        help="Transcription backend to use. Default is wav2vec.",
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base-960h",
        help="Wav2Vec2 model name for the transformers backend. Only used when --method=wav2vec.",
    )

    args = parser.parse_args()

    try:
        if args.method == "speech_recognition":
            transcription = transcribe_with_speech_recognition(args.audio)
        else:
            transcription = transcribe_with_wav2vec(args.audio, args.model)

        print("--- Transcription ---")
        print(transcription)
        print("---------------------")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
