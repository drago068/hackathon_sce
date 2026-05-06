import os
import tempfile
from functools import lru_cache
from pathlib import Path


WHISPER_MODEL_NAME = "base"


@lru_cache(maxsize=1)
def get_whisper_model(model_name: str = WHISPER_MODEL_NAME):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_uploaded_audio(
    uploaded_file,
    model_name: str = WHISPER_MODEL_NAME,
    language_code: str | None = None,
) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name or "audio.wav").suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        model = get_whisper_model(model_name)
        segments, _ = model.transcribe(
            temp_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return text.strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
