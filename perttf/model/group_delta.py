"""Observed-control log1p response head, separate from source reconstruction."""
import copy

import torch
from torch import nn, Tensor


class GroupDeltaDecoder(nn.Module):
    def __init__(self, model, rank: int):
        super().__init__()
        pert_dim = model.d_model if model.pert_dim is None else model.pert_dim
        self.shared = nn.Linear(pert_dim, model.d_model, bias=False)
        self.cell_projection = nn.Linear(model.d_model, rank)
        self.pert_projection = nn.Linear(pert_dim, rank, bias=False)
        self.adjustment = nn.Linear(rank, model.d_model, bias=False)
        self.gene2query = copy.deepcopy(model.mvc_decoder.gene2query)
        self.W = copy.deepcopy(model.mvc_decoder.W)
        nn.init.zeros_(self.shared.weight)
        nn.init.zeros_(self.adjustment.weight)

    def forward(self, z: Tensor, e: Tensor, genes: Tensor, source: Tensor) -> dict:
        query = self.W(torch.sigmoid(self.gene2query(genes)))
        shared = self.shared(e)
        adjustment = self.adjustment(torch.tanh(self.cell_projection(z)) * self.pert_projection(e))
        shared_expression = torch.bmm(query, shared.unsqueeze(2)).squeeze(2).float()
        adjustment_expression = torch.bmm(query, adjustment.unsqueeze(2)).squeeze(2).float()
        raw = source.float() + shared_expression + adjustment_expression
        return dict(delta_prediction=raw.clamp_min(0), shared_expression=shared_expression,
                    adjustment_expression=adjustment_expression, negative_fraction=(raw < 0).float().mean())
