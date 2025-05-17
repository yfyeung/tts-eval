import utmosv2
from mos_prediction.base import MOSModel


class UTMOSv2Model(MOSModel):
    def __init__(self, device=None):
        super().__init__(device)

        self.model = utmosv2.create_model(pretrained=True, device=device)

    def predict_mos(self, audio_file: str) -> float:
        return self.model.predict(input_path=audio_file, device=self.device)
