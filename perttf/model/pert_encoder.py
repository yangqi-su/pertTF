"""Fixed multi-source features with a gene-level trainable fallback."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _feature_table(table: tuple) -> Tuple[list, Tensor]:
    genes, values = table
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate perturbation IDs within a source")
    if len(genes) != values.shape[0]:
        raise ValueError("Source gene count must match embedding row count")
    features = torch.as_tensor(values, device="cpu", dtype=torch.float32).detach().clone()
    return list(genes), features


class PerturbationSource(nn.Module):
    """One fixed feature table and its trainable, normalized projection."""

    def __init__(self, features: Tensor, row_indices: Tensor, embedding_dim: int):
        super().__init__()
        self.register_buffer("features", features)
        self.register_buffer("row_indices", row_indices)
        self.proj = nn.Sequential(
            nn.Linear(features.shape[1], embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, features: Tensor) -> Tensor:
        return self.norm(self.proj(features))


class UnifiedPertEncoder(nn.Module):
    """Average available normalized sources, or learn a missing-gene vector.

    The supplied expanded mapping is shared by the dataset and classifier.
    The encoder only translates these indices into rows of each feature table.
    """

    def __init__(
        self,
        pert_sources: dict,
        genotype_to_index: Dict[str, int],
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        control_label: str = "WT",
    ):
        super().__init__()
        if not genotype_to_index or sorted(genotype_to_index.values()) != list(range(len(genotype_to_index))):
            raise ValueError("genotype_to_index must use contiguous indices starting at zero")
        if any(not isinstance(g, str) or not g for g in genotype_to_index):
            raise ValueError("Perturbation IDs must be nonempty strings")
        if control_label not in genotype_to_index:
            raise ValueError("The control label must be present in genotype_to_index")
        if padding_idx is not None and not 0 <= padding_idx < len(genotype_to_index):
            raise ValueError("pert_pad_id must be an index in genotype_to_index")
        if embedding_dim < 2:
            raise ValueError("UnifiedPertEncoder requires pert_dim >= 2 for LayerNorm")
        if any(not isinstance(name, str) or not name or '.' in name for name in pert_sources):
            raise ValueError("Source names must be nonempty strings without dots")
        self.control_label = control_label
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        self.perturbation_ids = set(genotype_to_index)
        self.source_genes = {}
        tables = {}
        for name, table in pert_sources.items():
            genes, features = _feature_table(table)
            missing = set(genes).difference(genotype_to_index)
            if missing:
                raise ValueError(f"Source {name!r} contains IDs absent from the supplied mapping: {sorted(missing)[:10]}; build the expanded mapping before model construction")
            if control_label in genes:
                features[genes.index(control_label)] = 0
            else:
                genes.append(control_label)
                features = torch.cat([features, features.new_zeros(1, features.shape[1])])
            self.source_genes[name] = genes
            tables[name] = features
        n_perturbations = len(genotype_to_index)
        covered = torch.zeros(n_perturbations, dtype=torch.bool)
        self.sources = nn.ModuleDict()
        for name, features in tables.items():
            rows = torch.full((n_perturbations,), -1, dtype=torch.long)
            for row, gene in enumerate(self.source_genes[name]):
                rows[genotype_to_index[gene]] = row
            covered |= rows >= 0
            self.sources[name] = PerturbationSource(features, rows, embedding_dim)
        # With no external sources, WT is still fixed zero before the final norm.
        covered[genotype_to_index[control_label]] = True
        fallback_indices = torch.full((n_perturbations,), -1, dtype=torch.long)
        fallback_indices[~covered] = torch.arange(int((~covered).sum()))
        self.register_buffer("fallback_indices", fallback_indices)
        self.fallback = nn.Embedding(int((~covered).sum()), embedding_dim)
        # One row per perturbation with no feature source -> sparse updates (see PertLabelEncoder).
        self.fallback.sparse_parameter_updates = True
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, indices: Tensor) -> Tensor:
        if torch.any(indices < 0) or torch.any(indices >= len(self.perturbation_ids)):
            raise ValueError("Perturbation index is outside the registered encoder mapping")
        flat = indices.reshape(-1)
        output = self.enc_norm.weight.new_zeros(flat.numel(), self.embedding_dim)
        counts = output.new_zeros(flat.numel(), 1)
        for source in self.sources.values():
            rows = source.row_indices[flat]
            available = rows >= 0
            if available.any():
                output[available] += source(source.features[rows[available]]).to(output.dtype)
                counts[available] += 1
        output = output / counts.clamp_min(1)
        fallback_rows = self.fallback_indices[flat]
        missing = fallback_rows >= 0
        if missing.any():
            output[missing] = self.fallback(fallback_rows[missing])
        output = self.enc_norm(output)
        if self.padding_idx is not None:
            output = output.masked_fill((flat == self.padding_idx).unsqueeze(-1), 0.0)
        return output.reshape(*indices.shape, self.embedding_dim)

    def encode_queries(self, query_sources: dict) -> Tuple[list, Tensor]:
        """Encode temporary new IDs with existing branches, without registering them.

        The caller chooses biological compatibility; only IDs, dimensions and
        row alignment are checked. Tables use the same (genes, matrix) format.
        """
        if self.training:
            raise RuntimeError("Temporary perturbation queries are supported only in eval mode")
        genes = []
        query_indices = {}
        tables = {}
        for name, table in query_sources.items():
            if name not in self.sources:
                raise ValueError(f"Unknown source type {name!r}; no trained projection exists")
            source_genes, features = _feature_table(table)
            if features.shape[1] != self.sources[name].features.shape[1]:
                raise ValueError(f"Query feature dimension does not match source {name!r}")
            for gene in source_genes:
                if gene in self.perturbation_ids:
                    raise ValueError(f"Query perturbation {gene!r} is already registered")
                if gene not in query_indices:
                    query_indices[gene] = len(genes)
                    genes.append(gene)
            tables[name] = (source_genes, features)
        if not genes:
            raise ValueError("Supply at least one temporary perturbation feature vector")
        output = self.enc_norm.weight.new_zeros(len(genes), self.embedding_dim)
        counts = output.new_zeros(len(genes), 1)
        for name, (source_genes, features) in tables.items():
            rows = torch.tensor([query_indices[g] for g in source_genes], device=output.device)
            features = features.to(device=output.device, dtype=output.dtype)
            output[rows] += self.sources[name](features).to(output.dtype)
            counts[rows] += 1
        return genes, self.enc_norm(output / counts)

    def source_config(self) -> dict:
        """Non-tensor reconstruction information; feature values live in state_dict."""
        return {
            "control_label": self.control_label,
            "sources": {
                name: {"genes": genes}
                for name, genes in self.source_genes.items()
            },
        }


_AGGREGATIONS = ("sum", "mean", "sqrt")


def reduce_perturbation_embeddings(emb: Tensor, labels: Tensor, aggregation: str) -> Tensor:
    """Reduce ``(batch, k, dim)`` embeddings to ``(batch, dim)``, ignoring ``-1`` pads."""
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {_AGGREGATIONS}, got {aggregation!r}")
    mask = (labels >= 0).unsqueeze(-1).to(emb.dtype)
    total = (emb * mask).sum(dim=1)
    if aggregation == "sum":
        return total
    n_valid = mask.sum(dim=1).clamp_min(1.0)
    if aggregation == "mean":
        return total / n_valid
    return total / n_valid.sqrt()


def encode_perturbation_set(encoder: nn.Module, labels: Tensor, aggregation: str = "sum") -> Tensor:
    """Encode scalar labels ``(batch,)`` or padded sets ``(batch, k)`` into ``(batch, dim)``."""
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {_AGGREGATIONS}, got {aggregation!r}")
    if labels.dim() == 1:
        return encoder(labels)
    emb = encoder(labels.clamp_min(0))
    return reduce_perturbation_embeddings(emb, labels, aggregation)


class KnockoutLookup(nn.Module):
    """A knockout's representation as a weighted average of learned codes: one per table knockout (those with training
    targets) and one "no match" code. Weights are a softmax over [no-match score, sharpness * cos(query, table key)],
    with query and keys the base encoder's embeddings. In training, each knockout present in a batch is hidden from its
    own table row with probability hide, so the sharpness, the no-match score and code learn how to represent a knockout
    that has no row; at evaluation a table knockout finds its own row and any other knockout has none.
    With self_bonus, a knockout's own row also gets a learned bonus delta added to its score, so a table knockout can
    find itself whatever the sharpness: the learned projection can place an unseen knockout as close to a table
    knockout as that knockout is to itself (MODELLING_DISCUSSION.md section 61), which cos alone cannot tell apart.
    With hard_identity, a table knockout that is not hidden is represented by exactly its own code, and the weighted
    average (which then never includes the knockout's own row) serves only hidden knockouts and knockouts without a row.
    """

    def __init__(self, table: list, n_pert: int, dim: int, hide: float, self_bonus: bool = False,
                 hard_identity: bool = False):
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.long))
        row = torch.full((n_pert,), -1, dtype=torch.long)
        row[self.table] = torch.arange(len(table))
        self.register_buffer("row_of", row)
        # Unit-scale codes, like the base encoder's LayerNorm output; the no-match code starts at 0 (no knockout effect).
        self.codes = nn.Parameter(torch.randn(len(table), dim))
        self.null_code = nn.Parameter(torch.zeros(dim))
        self.null_score = nn.Parameter(torch.zeros(()))
        self.log_sharpness = nn.Parameter(torch.tensor(2.0))   # sharpness e^2 ~ 7.4
        self.self_bonus = nn.Parameter(torch.tensor(2.0)) if self_bonus else None
        self.hard_identity = hard_identity
        self.hide = hide

    def hidden(self, labels: Tensor) -> Tensor:
        """(batch,) bool: the pair's knockout is hidden from its own row in this training step (all pairs of a knockout
        together, each table knockout present with probability hide); all False at evaluation."""
        own = self.row_of[labels]
        if not (self.training and self.hide > 0):
            return torch.zeros_like(own, dtype=torch.bool)
        present = torch.unique(labels[own >= 0])
        return torch.isin(labels, present[torch.rand(len(present), device=labels.device) < self.hide])

    def weights(self, query: Tensor, keys: Tensor, labels: Tensor, exclude_own: Optional[Tensor] = None) -> Tensor:
        """(batch, 1 + table) softmax weights; column 0 is the no-match code. Rows with exclude_own (default: hidden)
        get no weight on their own row."""
        sim = torch.exp(self.log_sharpness) * nn.functional.normalize(query, dim=-1) @ nn.functional.normalize(keys, dim=-1).T
        own = self.row_of[labels]
        if self.self_bonus is not None:
            is_own = nn.functional.one_hot(own.clamp(min=0), len(self.table)) * (own >= 0).unsqueeze(1)
            sim = sim + self.self_bonus * is_own
        scores = torch.cat([self.null_score.expand(len(query), 1), sim], dim=1)
        if exclude_own is None:
            exclude_own = self.hidden(labels)
        rows = torch.nonzero(exclude_own & (own >= 0)).squeeze(1)
        scores[rows, 1 + own[rows]] = float("-inf")
        return torch.softmax(scores, dim=1)

    def forward(self, query: Tensor, keys: Tensor, labels: Tensor) -> Tensor:
        hidden = self.hidden(labels)
        if not self.hard_identity:
            w = self.weights(query, keys, labels, exclude_own=hidden)
            return w[:, :1] * self.null_code + w[:, 1:] @ self.codes
        own = self.row_of[labels]
        w = self.weights(query, keys, labels, exclude_own=own >= 0)
        blend = w[:, :1] * self.null_code + w[:, 1:] @ self.codes
        use_own = (own >= 0) & ~hidden
        return torch.where(use_own.unsqueeze(1), self.codes[own.clamp(min=0)], blend)
