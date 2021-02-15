from typing import List, Optional

import torch
from torch import Tensor, nn
from yukarin_sos.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        accent_embedding_size: int,
        hidden_size_list: List[int],
        kernel_size_list: List[int],
    ):
        layer_num = len(hidden_size_list)
        assert len(kernel_size_list) == layer_num

        super().__init__()

        self.phoneme_size = phoneme_size
        self.phoneme_padding_index = phoneme_size

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size,
            embedding_dim=phoneme_embedding_size,
        )
        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if speaker_size > 0
            else None
        )

        self.start_accent_embedder = (
            nn.Embedding(
                num_embeddings=2,
                embedding_dim=accent_embedding_size,
            )
            if accent_embedding_size > 0
            else None
        )
        self.end_accent_embedder = (
            nn.Embedding(
                num_embeddings=2,
                embedding_dim=accent_embedding_size,
            )
            if accent_embedding_size > 0
            else None
        )

        input_size = (
            phoneme_embedding_size + speaker_embedding_size + accent_embedding_size * 2
        )

        convs: List[nn.Module] = []
        for i in range(layer_num):
            convs.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=(hidden_size_list[i - 1] if i > 0 else input_size),
                        out_channels=hidden_size_list[i],
                        kernel_size=kernel_size_list[i],
                        padding=kernel_size_list[i] // 2,
                    )
                )
            )
            convs.append(nn.SiLU(inplace=True))

        self.convs = nn.Sequential(*convs)

        self.post = nn.Conv1d(hidden_size_list[-1], 2, kernel_size=1)

    def forward(
        self,
        phoneme: Tensor,  # (batch, length)
        start_accent: Optional[Tensor],  # (batch, length)
        end_accent: Optional[Tensor],  # (batch, length)
        speaker_id: Optional[Tensor],  # (batch, )
    ):
        h = self.phoneme_embedder(phoneme)  # (batch, length, ?)
        h = h.transpose(1, 2)  # (batch, ?, length)

        if self.speaker_embedder is not None:
            speaker_id = self.speaker_embedder(speaker_id)  # (batch, ?)
            speaker_id = speaker_id.unsqueeze(2)  # (batch, ?, 1)
            speaker = speaker_id.expand(
                speaker_id.shape[0], speaker_id.shape[1], h.shape[2]
            )  # (batch, ?, length)
            h = torch.cat((h, speaker), dim=1)  # (batch, ?, length)

        if (
            self.start_accent_embedder is not None
            and self.end_accent_embedder is not None
        ):
            start_accent = self.start_accent_embedder(
                start_accent
            )  # (batch, length, ?)
            end_accent = self.end_accent_embedder(end_accent)  # (batch, length, ?)
            accent = torch.cat((start_accent, end_accent), dim=2)  # (batch, length, ?)
            accent = accent.transpose(1, 2)  # (batch, ?, length)
            h = torch.cat((h, accent), dim=1)  # (batch, ?, length)

        h = self.convs(h)  # (batch, ?, length)
        h = self.post(h)  # (batch, 2, length)
        f0, vuv = h[:, 0], h[:, 1]  # (batch, length)
        return dict(f0=f0, vuv=vuv)


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        accent_embedding_size=config.accent_embedding_size,
        hidden_size_list=config.hidden_size_list,
        kernel_size_list=config.kernel_size_list,
    )
