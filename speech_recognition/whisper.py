import torch
import whisper
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from speech_recognition.base import ASRModel


class WhisperASRModel(ASRModel):
    def __init__(
        self,
        version="large-v3",
        language="en",
        device=None,
    ):
        super().__init__(device)

        self.model = whisper.load_model(version, device=self.device)
        self.language = language
        self.tn = (
            EnglishTextNormalizer() if self.language == "en" else BasicTextNormalizer()
        )

    def transcribe(self, audio: torch.Tensor) -> str:
        return self.model.transcribe(
            audio.squeeze(0), language=self.language, beam_size=1
        )["text"]

    def text_normalize(self, text: str) -> str:
        return self.tn(text)
