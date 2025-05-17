from abc import ABC, abstractmethod

import torch


class MOSModel(ABC):
    """
    Abstract base class for naturalness mean opinion score (MOS) prediction models.
    """

    def __init__(self, device: torch.device):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    @abstractmethod
    def predict_mos(self, audio_path: str) -> float:
        pass
