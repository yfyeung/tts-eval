from abc import ABC, abstractmethod

import torch


class ASVModel(ABC):
    """
    Abstract base class for Automatic Speaker Verification (ASV) models.
    """

    def __init__(self, device: torch.device):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    def compute_similarity(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> float:
        """
        Compute cosine similarity between two audio samples.
        :param source_audio: Tensor representing the source audio
        :param target_audio: Tensor representing the target audio
        :param device: Torch device (e.g., 'cpu' or 'cuda')
        :return: Similarity score (float)
        """
        if target_audio.shape[-1] < 1600:
            return 0.0

        source_emb = self.model(source_audio.to(self.device))
        target_emb = self.model(target_audio.to(self.device))

        return (
            torch.nn.functional.cosine_similarity(source_emb, target_emb).cpu().item()
        )
