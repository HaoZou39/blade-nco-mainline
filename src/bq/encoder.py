"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

Migrated into BLADE-NCO. Encoder-layer residual path is ReZero when
batchnorm=False, matching the official TSP checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear
from torch.nn.modules import Dropout
from torch.nn.parameter import Parameter

from .attention import MultiHeadAttention


class ResidualNorm(nn.Module):
    def __init__(self, emb_size, dropout, batchnorm):
        super().__init__()
        self.batchnorm = batchnorm
        self.norm = BatchNorm1d(emb_size, track_running_stats=False) if batchnorm else None
        self.alpha = 1.0 if self.batchnorm else Parameter(torch.tensor(0.0))
        self.dropout = Dropout(dropout)

    def forward(self, state_before, modified_after):
        state_before = state_before + self.alpha * self.dropout(modified_after)
        if self.batchnorm:
            state_before = state_before.transpose(2, 1)
            state_before = self.norm(state_before)
            return state_before.transpose(2, 1)
        return state_before


class EncoderLayer(nn.Module):
    def __init__(self, nb_heads, activation_attention, emb_size, dim_ff, activation_ff, dropout, batchnorm):
        super().__init__()
        self.attn = MultiHeadAttention(emb_size, nb_heads, activation=activation_attention)
        self.linear1 = Linear(emb_size, dim_ff)
        self.linear2 = Linear(dim_ff, emb_size)
        if activation_ff == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_ff == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError(f"unsupported feed-forward activation: {activation_ff}")
        self.res_norm1 = ResidualNorm(emb_size, dropout, batchnorm)
        self.res_norm2 = ResidualNorm(emb_size, dropout, batchnorm)

    def forward(self, state):
        state_rc = state
        state = self.attn(state, state)
        state = self.res_norm1(state_rc, state)
        state_rc = state
        state = self.linear2(self.activation(self.linear1(state)))
        return self.res_norm2(state_rc, state)
