from abc import ABC, abstractmethod

import editdistance
import torch


class ASRModel(ABC):
    def __init__(self, device: torch.device):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    @abstractmethod
    def transcribe(self, audio: torch.Tensor) -> str:
        pass

    @abstractmethod
    def text_normalize(self, text: str) -> str:
        pass

    def compute_wer(self, hyp: str, ref: str) -> float:
        error = editdistance.eval(hyp.split(), ref.split())
        total = len(ref.split())
        return {
            "error": error,
            "total": total,
            "wer": error / total if total > 0 else 0.0,
        }
