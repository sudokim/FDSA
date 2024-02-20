from typing import List, Union, Optional

import torch
from torch import nn

from .feature_based import FeatureBasedEncoder
from .item_based import ItemBasedEncoder


class FDSA(nn.Module):
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

        self.feature_based = FeatureBasedEncoder(
            num_items=num_items,
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pretrained=pretrained,
        )

        self.item_based = ItemBasedEncoder(
            num_items=num_items,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

        self.feedforward = nn.Linear(d_model * 2, d_model, bias=True)

    def forward(self, item_seq: torch.Tensor, feature_seq: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # item_seq: (batch_size, num_items)
        # feature_seq: (batch_size, num_items, num_features)
        # mask: (batch_size, num_items) True for padding

        if mask is None:
            mask = torch.eq(item_seq, 0)  # (batch_size, num_items)

        sequence_length = torch.eq(mask, False).sum(dim=-1)  # (batch_size,)
        batch_index = torch.arange(sequence_length.size(0))

        item_hidden_state = self.item_based.forward(item_seq, mask)  # (batch_size, num_items, d_model)
        feature_hidden_state = self.feature_based.forward(feature_seq, mask)  # (batch_size, num_items, d_model)

        item_output = item_hidden_state[batch_index, sequence_length - 1]  # (batch_size, d_model)
        feature_output = feature_hidden_state[batch_index, sequence_length - 1]  # (batch_size, d_model)

        representation = torch.cat((item_output, feature_output), dim=-1)  # (batch_size, d_model * 2)
        representation = self.feedforward.forward(representation)  # (batch_size, d_model)

        logit = torch.matmul(representation, self.item_based.embedding.weight.T)  # (batch_size, num_items)

        return logit
