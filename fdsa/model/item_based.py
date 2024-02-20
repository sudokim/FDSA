import torch
from torch import nn


class ItemBasedEmbedding(nn.Module):
    def __init__(self, num_items: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, num_items)
        return self.embedding.forward(x)


class ItemBasedAttentionBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()

        self_attn_block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.TransformerEncoder(self_attn_block, num_layers=num_layers)
        self.position_embedding = nn.Embedding(max_len + 1, d_model, padding_idx=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (batch_size, num_items, d_embed)
        # mask: (batch_size, num_items) True for padding
        batch_size = x.size(0)
        num_items = x.size(1)

        position = (
            torch.arange(1, num_items + 1).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        )  # (batch_size, num_items)
        position[mask] = 0
        position_embedding = self.position_embedding.forward(position)  # (batch_size, num_items, d_embed)
        input_matrix = x + position_embedding

        output = self.self_attn.forward(
            input_matrix, src_key_padding_mask=mask, is_causal=True
        )  # (batch_size, num_items, d_embed)

        return output


class ItemBasedEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()

        self.embedding = ItemBasedEmbedding(num_items, d_model)
        self.attention = ItemBasedAttentionBlock(num_layers, d_model, nhead, dim_feedforward, dropout, max_len)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        # seq: (batch_size, num_items)
        # mask: (batch_size, num_items) True for padding
        x = self.embedding.forward(seq)

        return self.attention.forward(x, mask)
