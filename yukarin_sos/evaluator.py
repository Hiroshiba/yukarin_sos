from typing import Optional

import numpy
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_sos.generator import Generator


class GenerateEvaluator(nn.Module):
    def __init__(
        self,
        generator: Generator,
    ):
        super().__init__()
        self.generator = generator

    def forward(
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

        out_f0 = self.generator.generate(
            phoneme=phoneme,
            start_accent=start_accent,
            end_accent=end_accent,
            speaker_id=speaker_id,
        )
        out_f0 = out_f0[~padded.cpu().numpy()]
        out_vuv = out_f0 != 0

        in_f0 = f0[~padded].cpu().numpy()
        in_vuv = in_f0 != 0

        vuv = numpy.bitwise_and(out_vuv, in_vuv)

        f0_diff = numpy.abs(out_f0[vuv] - in_f0[vuv]).mean()
        vuv_acc = (out_vuv == in_vuv).mean()

        scores = {"f0_diff": (f0_diff, batch_size), "vuv_acc": (vuv_acc, batch_size)}

        report(scores, self)
        return scores
