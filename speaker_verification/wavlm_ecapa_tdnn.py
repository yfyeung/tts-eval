import logging
import urllib.request
from pathlib import Path

import torch

from speaker_verification.base import ASVModel
from speaker_verification.models.ecapa_tdnn import ECAPA_TDNN

MODEL_URL = "https://huggingface.co/yfyeung/wavlm-large-speaker-verification/resolve/main/wavlm-large.pt"
CACHE_PATH = Path(
    "~/.cache/tts_eval/download/wavlm-large-speaker-verification/wavlm-large.pt"
).expanduser()


class WavlmECAPAASVModel(ASVModel):
    def __init__(self, device=None):
        super().__init__(device)

        self.model = ECAPA_TDNN(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            feat_type="wavlm_large",
            sr=16000,
            feature_selection="hidden_states",
            update_extract=False,
            config_path=None,
        )

        model_path = self.download_model(CACHE_PATH, MODEL_URL)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model"], strict=False
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def download_model(model_path: Path, url: str) -> Path:
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Downloading model from {url} ...")
            urllib.request.urlretrieve(url, model_path)
            logging.info("Download complete.")
        return model_path
