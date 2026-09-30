"""Small BERT-style product-history encoder trained from scratch, not a language model."""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertConfig, BertModel


class ProductHistoryTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, sequence_length: int, num_products: int, hidden_size: int = 64
    ):
        super().__init__()
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four attention heads")
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=hidden_size * 2,
            max_position_embeddings=sequence_length + 2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.type_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.numeric_projection = nn.Linear(2, hidden_size)
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.head = nn.Linear(hidden_size, num_products)

    def forward(
        self, event_types: torch.Tensor, numeric: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = self.type_embedding(event_types) + self.numeric_projection(numeric)
        encoded = self.encoder(inputs_embeds=token_embeddings, attention_mask=attention_mask)
        return self.head(encoded.last_hidden_state[:, 0])
