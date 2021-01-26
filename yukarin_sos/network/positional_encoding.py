import math

import numpy
import torch
from torch import nn
from torch.functional import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, default_length: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.initialized = False
        self.recreate_pe(length=default_length)

    def recreate_pe(self, length: int):
        if not self.initialized:
            device = "cpu"
            self.initialized = True
        else:
            device = self.pe.device

        hidden_size = self.hidden_size
        length = 2 ** int(numpy.ceil(numpy.log2(length)))

        pe = torch.zeros(length, hidden_size)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        pe = pe.to(device)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor):
        length = x.shape[0]
        if length > self.pe.shape[0]:
            self.recreate_pe(length=length)

        x = x + self.pe[:length].expand_as(x)
        return x
