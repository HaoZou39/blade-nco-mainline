"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

Migrated into BLADE-NCO. Constructor signature is byte-compatible with the
official TSP checkpoint loader used by the validated fine-tuning protocol:

    BQModel(2, 192, 512, "relu", 9, 12, "softmax", 0.0, False, "tsp")
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import Embedding, Linear, Module
from torch.nn.modules import ModuleList

from .encoder import EncoderLayer


class BQModel(Module):
    def __init__(
        self,
        dim_input_nodes,
        emb_size,
        dim_ff,
        activation_ff,
        nb_layers_encoder,
        nb_heads,
        activation_attention,
        dropout,
        batchnorm,
        problem="tsp",
    ):
        assert problem in {"tsp", "cvrp", "kp", "op"}
        super().__init__()
        self.problem = problem
        self.input_emb = Linear(dim_input_nodes, emb_size)
        if problem != "kp":
            self.begin_end_tokens = Embedding(2, emb_size)
        self.nb_layers_encoder = nb_layers_encoder
        self.encoder = ModuleList(
            [
                EncoderLayer(
                    nb_heads,
                    activation_attention,
                    emb_size,
                    dim_ff,
                    activation_ff,
                    dropout,
                    batchnorm,
                )
                for _ in range(nb_layers_encoder)
            ]
        )
        output_dim = 2 if problem == "cvrp" else 1
        self.scores_projection = Linear(emb_size, output_dim)

    def forward(self, inputs, **problem_data):
        if self.problem == "op":
            assert "dist_matrices" in problem_data
        input_emb = self.input_emb(inputs)
        if self.problem != "kp":
            input_emb[:, 0, :] += self.begin_end_tokens(torch.tensor([[0]], device=inputs.device)).squeeze(1)
            input_emb[:, -1, :] += self.begin_end_tokens(torch.tensor([[1]], device=inputs.device)).squeeze(1)
        state = input_emb
        for layer in self.encoder:
            state = layer(state)
        scores = self.scores_projection(state).squeeze(-1)
        if self.problem == "tsp":
            scores[:, 0] = scores[:, -1] = -np.inf
        elif self.problem == "cvrp":
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            scores[..., 0][problem_data["demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
        elif self.problem == "op":
            scores[:, 0] = scores[:, -1] = -np.inf
            scores[
                problem_data["dist_matrices"][:, 0]
                + problem_data["dist_matrices"][:, -1]
                - inputs[..., 3]
                > 1e-6
            ] = -np.inf
        elif self.problem == "kp":
            scores[problem_data["weights"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
        return scores.reshape(scores.shape[0], -1)
