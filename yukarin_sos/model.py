from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_sos.config import ModelConfig
from yukarin_sos.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def __call__(
        self,
        f0: Tensor,
        phoneme: Tensor,
        silence: Tensor,
        start_accent: Tensor,
        end_accent: Tensor,
        padded: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = len(f0)

        d = self.predictor(
            phoneme=phoneme,
            start_accent=start_accent,
            end_accent=end_accent,
            f0=f0.roll(1, dims=1),
            speaker_id=speaker_id,
        )
        output_f0 = d["f0"][~padded]
        output_vuv = d["vuv"][~padded]

        f0 = f0[~padded]
        vuv = f0 != 0

        loss_f0 = F.l1_loss(output_f0[vuv], f0[vuv])
        loss_vuv = F.binary_cross_entropy_with_logits(output_vuv, vuv.to(torch.float32))

        loss_f0 = loss_f0 * self.model_config.f0_loss_weight
        loss_vuv = loss_vuv * self.model_config.vuv_loss_weight
        loss = loss_f0 + loss_vuv

        # report
        values = dict(
            loss=loss,
            loss_f0=loss_f0,
            loss_vuv=loss_vuv,
        )
        if not self.training:
            weight = batch_size
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
