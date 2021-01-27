from pathlib import Path
from typing import Optional, Union

import numpy
import torch

from yukarin_sos.config import Config
from yukarin_sos.network.predictor import Predictor, create_predictor


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool = True,
    ):
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        phoneme: Union[numpy.ndarray, torch.Tensor],
        speaker_id: Optional[Union[numpy.ndarray, torch.Tensor]],
    ):
        if isinstance(phoneme, numpy.ndarray):
            phoneme = torch.from_numpy(phoneme)
        phoneme = phoneme.to(self.device)

        if speaker_id is not None:
            if isinstance(speaker_id, numpy.ndarray):
                speaker_id = torch.from_numpy(speaker_id)
            speaker_id = speaker_id.to(torch.int64).to(self.device)

        with torch.no_grad():
            d = self.predictor(
                phoneme=phoneme,
                speaker_id=speaker_id,
            )
        f0, vuv = d["f0"], d["vuv"]
        f0[vuv < 0] = 0
        return f0.cpu().numpy()
