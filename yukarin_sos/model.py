from typing import Optional, Sequence

import numpy
import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_sos.config import ModelConfig
from yukarin_sos.network.predictor import Predictor


def accuracy(output: Tensor, target: Tensor):
    with torch.no_grad():
        if output.shape[1] > 1:
            indexes = torch.argmax(output, dim=1)
            correct = torch.eq(indexes, target)
        else:
            correct = torch.eq(output >= 0, target)
        return correct.float().mean()


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def __call__(
        self,
        f0: Sequence[Tensor],
        phoneme: Sequence[Tensor],
        phoneme_list: Sequence[Tensor],
        speaker_id: Optional[Sequence[Tensor]] = None,
    ):
        batch_size = len(f0)

        speaker_id = torch.stack(speaker_id)

        d = self.predictor(
            f0=[h[:-1] for h in f0],
            phoneme=[h[:-1] for h in phoneme],
            phoneme_list=phoneme_list,
            speaker_id=speaker_id,
        )
        output_f0 = d["f0"]
        output_phoneme = d["phoneme"]
        output_vuv = d["vuv"]
        output_stop = d["stop"]

        stacked_f0 = torch.cat(f0)
        stacked_phoneme = torch.cat(phoneme)
        stacked_vuv = stacked_f0 != 0

        stacked_stop = torch.zeros_like(stacked_f0, dtype=torch.bool)
        stacked_stop[numpy.cumsum([h.shape[0] for h in f0]) - 1] = True

        # loss
        loss_f0 = F.l1_loss(output_f0[stacked_vuv], stacked_f0[stacked_vuv])
        loss_phoneme = F.cross_entropy(output_phoneme, stacked_phoneme)
        loss_vuv = F.binary_cross_entropy_with_logits(
            output_vuv, stacked_vuv.to(torch.float32)
        )
        loss_stop = F.binary_cross_entropy_with_logits(
            output_stop,
            stacked_stop.to(torch.float32),
            pos_weight=torch.ones_like(output_stop) * 10,
        )

        loss_f0 = loss_f0 * self.model_config.f0_loss_weight
        loss_phoneme = loss_phoneme * self.model_config.phoneme_loss_weight
        loss_vuv = loss_vuv * self.model_config.vuv_loss_weight
        loss_stop = loss_stop * self.model_config.stop_loss_weight
        loss = loss_f0 + loss_phoneme + loss_vuv + loss_stop

        # metric
        accuracy_phoneme = accuracy(output_phoneme, stacked_phoneme)
        accuracy_vuv = accuracy(output_vuv, stacked_vuv)
        accuracy_stop = accuracy(output_stop, stacked_stop)

        # report
        values = dict(
            loss=loss,
            loss_f0=loss_f0,
            loss_phoneme=loss_phoneme,
            loss_vuv=loss_vuv,
            loss_stop=loss_stop,
            accuracy_phoneme=accuracy_phoneme,
            accuracy_vuv=accuracy_vuv,
            accuracy_stop=accuracy_stop,
        )
        if not self.training:
            weight = batch_size
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
