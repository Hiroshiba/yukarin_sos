from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from yukarin_sos.config import NetworkConfig
from yukarin_sos.network.positional_encoding import PositionalEncoding


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        transformer_hidden_size: int,
        tranformer_head_num: int,
        transformer_encoder_layer_num: int,
        transformer_decoder_layer_num: int,
        tranformer_linear_size: int,
    ):
        super().__init__()

        self.with_speaker = speaker_size > 0
        self.phoneme_size = phoneme_size
        self.phoneme_padding_index = phoneme_size

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size + 1,
            embedding_dim=phoneme_embedding_size,
            padding_idx=self.phoneme_padding_index,
        )
        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if self.with_speaker
            else None
        )

        self.source_positional_encoding = PositionalEncoding(
            hidden_size=phoneme_embedding_size
        )
        self.source_pre = nn.Linear(
            phoneme_embedding_size
            + (speaker_embedding_size if self.with_speaker else 0),
            transformer_hidden_size,
        )

        self.target_size = 1 + phoneme_embedding_size  # f0 + phoneme
        self.target_positional_encoding = PositionalEncoding(
            hidden_size=phoneme_embedding_size
        )
        self.target_pre = nn.Linear(
            self.target_size + (speaker_embedding_size if self.with_speaker else 0),
            transformer_hidden_size,
        )

        self.transformer = nn.Transformer(
            d_model=transformer_hidden_size,
            nhead=tranformer_head_num,
            num_encoder_layers=transformer_encoder_layer_num,
            num_decoder_layers=transformer_decoder_layer_num,
            dim_feedforward=tranformer_linear_size,
        )

        self.post = nn.Linear(
            in_features=transformer_hidden_size,
            out_features=1 + phoneme_size + 1 + 1,  # f0 + phoneme + vuv + stop
        )

    def forward(
        self,
        f0: Sequence[Tensor],
        phoneme: Sequence[Tensor],
        phoneme_list: Sequence[Tensor],
        speaker_id: Optional[Tensor],
    ):
        source = pad_sequence(phoneme_list, padding_value=self.phoneme_padding_index)
        source = self.phoneme_embedder(source)
        source = self.source_positional_encoding(source)
        source_padding_mask = pad_sequence(
            [torch.zeros(p.shape[0], dtype=torch.bool) for p in phoneme_list],
            padding_value=True,
        ).T.to(source.device)

        target_f0 = pad_sequence(f0)
        target_phoneme = pad_sequence(phoneme, padding_value=self.phoneme_padding_index)
        target_phoneme = self.phoneme_embedder(target_phoneme)
        target_phoneme = self.target_positional_encoding(target_phoneme)
        target = torch.cat((target_f0, target_phoneme), dim=2)
        target = F.pad(target, (0, 0, 0, 0, 1, 0))
        target_padding_mask = pad_sequence(
            [torch.zeros(f.shape[0] + 1, dtype=torch.bool) for f in f0],
            padding_value=True,
        ).T.to(source.device)

        if self.with_speaker:
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=0)  # (1, batch_size, ?)

            speaker = speaker_id.expand(
                source.shape[0], speaker_id.shape[1], speaker_id.shape[2]
            )
            source = torch.cat((source, speaker), dim=2)

            speaker = speaker_id.expand(
                target.shape[0], speaker_id.shape[1], speaker_id.shape[2]
            )
            target = torch.cat((target, speaker), dim=2)

        autoregressive_mask = self.transformer.generate_square_subsequent_mask(
            target.shape[0]
        ).to(target.device)

        source = self.source_pre(source)
        target = self.target_pre(target)

        h = self.transformer(
            src=source,
            tgt=target,
            tgt_mask=autoregressive_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        output = self.post(h)
        output = output[~target_padding_mask.T]
        output_f0, output_phoneme, output_vuv, output_stop = torch.split(
            output, [1, self.phoneme_size, 1, 1], dim=1
        )
        return dict(
            f0=output_f0,
            phoneme=output_phoneme,
            vuv=output_vuv,
            stop=output_stop,
        )

    @torch.no_grad()
    def generate(
        self,
        phoneme_list: Tensor,  # (source_length, )
        speaker_id: Optional[Tensor],  # (1, )
        max_length: int,
    ):
        output_f0: List[Tensor] = []
        output_phoneme: List[Tensor] = []

        source = phoneme_list.unsqueeze(1)  # (source_length, 1)
        source = self.phoneme_embedder(source)  # (source_length, 1, ?)
        source = self.source_positional_encoding(source)  # (source_length, 1, ?)

        target = torch.zeros(
            (1, 1, self.target_size), dtype=torch.float32, device=phoneme_list.device
        )  # (1, 1, ?)

        if self.with_speaker:
            speaker_id = speaker_id.unsqueeze(0)  # (1, 1)
            speaker_id = self.speaker_embedder(speaker_id)  # (1, 1, ?)

            speaker = speaker_id.expand(
                source.shape[0], speaker_id.shape[1], speaker_id.shape[2]
            )  # (source_length, 1, ?)
            source = torch.cat((source, speaker), dim=2)  # (source_length, 1, ?)

        source = self.source_pre(source)  # (source_length, 1, ?)
        memory = self.transformer.encoder(source)  # (source_length, 1, ?)

        for _ in range(max_length):
            h = target
            if self.with_speaker:
                speaker = speaker_id.expand(
                    h.shape[0], speaker_id.shape[1], speaker_id.shape[2]
                )  # (target_length, 1, ?)
                h = torch.cat((h, speaker), dim=2)  # (target_length, 1, ?)

            h = self.target_pre(h)  # (target_length, 1, ?)

            h = self.transformer.decoder(tgt=h, memory=memory)  # (target_length, 1, ?)
            output = self.post(h)  # (target_length, 1, ?)

            output = output[-1:]  # (1, 1, ?)
            f0, phoneme, vuv, stop = torch.split(
                output, [1, self.phoneme_size, 1, 1], dim=2
            )  # (1, 1, ?)
            f0[vuv < 0] = 0
            phoneme = phoneme.argmax(dim=2)

            output_f0.append(f0.squeeze())
            output_phoneme.append(phoneme.squeeze())

            if stop > 0:
                break

            target_phoneme = self.phoneme_embedder(phoneme)  # (target_length, 1, ?)
            target_phoneme = self.target_positional_encoding(
                target_phoneme
            )  # (target_length, 1, ?)
            new_target = torch.cat((f0, target_phoneme), dim=2)  # (1, 1, ?)
            target = torch.cat((target, new_target), dim=0)  # (target_length, 1, ?)

        return dict(
            f0=torch.stack(output_f0),
            phoneme=torch.stack(output_phoneme),
        )


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        transformer_hidden_size=config.transformer_hidden_size,
        tranformer_head_num=config.tranformer_head_num,
        transformer_encoder_layer_num=config.transformer_encoder_layer_num,
        transformer_decoder_layer_num=config.transformer_decoder_layer_num,
        tranformer_linear_size=config.tranformer_linear_size,
    )
