import torch
from transformers import HubertForCTC, Wav2Vec2Processor

from speech_recognition.base import ASRModel


class HubertASRModel(ASRModel):
    def __init__(self, device=None):
        super().__init__(device)

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(
            self.device
        )

    def transcribe(self, audio: torch.Tensor) -> str:
        input = self.processor(
            audio.squeeze(0), sampling_rate=16000, return_tensors="pt"
        )
        input = {k: v.to(self.device) for k, v in input.items()}

        logits = self.model(**input).logits
        text = self.processor.decode(torch.argmax(logits, dim=-1)[0])

        return text

    def text_normalize(self, text: str) -> str:
        normalized_text = " ".join(
            "".join([i if i.isalpha() or i == "'" else " " for i in text]).split()
        ).upper()

        return normalized_text
