from typing import List, Union, Optional

import torch
from torch import nn


class FeatureBasedEmbedding(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_features: int,
        d_model: int,
        pretrained: Optional[List[Union[torch.Tensor, None]]] = None,
    ):
        super().__init__()

        if pretrained is None:
            pretrained = [None] * num_features

        assert len(pretrained) == num_features

        modules = []
        for p in pretrained:
            if p is None:
                modules.append(nn.Embedding(num_items + 1, d_model, padding_idx=0))
            else:
                assert isinstance(p, torch.Tensor)
                assert p.shape == (num_items + 1, d_model)
                modules.append(nn.Embedding.from_pretrained(p, padding_idx=0, freeze=True))

        self.embeddings = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, num_items, num_features)
        return torch.stack([emb.forward(x[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=2)


class VanillaAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_features: int):
        super().__init__()

        self.attn_linear = nn.Linear(d_model, num_features, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, num_items, num_features, d_embed)
        attr_logit = self.attn_linear(x)  # (batch_size, num_items, num_features)
        attr_weight = torch.softmax(attr_logit, dim=-1)  # (batch_size, num_items, num_features)

        feature_repr = torch.sum(torch.mul(x, attr_weight.unsqueeze(-1)), dim=2)  # (batch_size, num_items, d_embed)

        return feature_repr

    @classmethod
    def factory(cls, d_model: int, num_features: int):
        if num_features == 1:
            # (batch_size, num_items, 1, d_embed) -> (batch_size, num_items, d_embed)
            return nn.Flatten(start_dim=2, end_dim=-1)
        else:
            return cls(d_model, num_features)


class FeatureBasedAttentionBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_features: int,
        max_len: int,
    ):
        super().__init__()

        self.input_attn_layer = VanillaAttentionLayer.factory(d_model, num_features)
        self_attn_block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.TransformerEncoder(self_attn_block, num_layers=num_layers)
        self.position_embedding = nn.Embedding(max_len + 1, d_model, padding_idx=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (batch_size, num_items, num_features, d_embed)
        # mask: (batch_size, num_items) True for padding
        batch_size = x.size(0)
        num_items = x.size(1)

        feature_repr = self.input_attn_layer.forward(x)  # (batch_size, num_items, d_embed)
        position = (
            torch.arange(1, num_items + 1).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        )  # (batch_size, num_items)
        position[mask] = 0
        position_embedding = self.position_embedding.forward(position)  # (batch_size, num_items, d_embed)
        input_matrix = feature_repr + position_embedding

        output = self.self_attn.forward(
            input_matrix, src_key_padding_mask=mask, is_causal=True
        )  # (batch_size, num_items, d_embed)

        return output


class FeatureBasedEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_features: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        pretrained: Optional[List[Union[torch.Tensor, None]]] = None,
    ):
        super().__init__()

        self.embedding = FeatureBasedEmbedding(
            num_items=num_items, num_features=num_features, d_model=d_model, pretrained=pretrained
        )
        self.attention_block = FeatureBasedAttentionBlock(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_features=num_features,
            max_len=max_len,
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        # seq: (batch_size, num_items, num_features)
        # mask: (batch_size, num_items) True for padding
        x = self.embedding.forward(seq)

        return self.attention_block.forward(x, mask)
