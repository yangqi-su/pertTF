"""Fitting-only context/target means and uniformly sampled target-group blocks."""
import numpy as np
import torch
from scipy import sparse


class ContextKOBatchSampler:
    def __init__(self, group_ids, steps: int, groups_per_batch: int, cells_per_group: int, seed: int):
        self.pools = [np.flatnonzero(group_ids == group) for group in np.unique(group_ids)]
        self.steps = steps
        self.groups_per_batch = groups_per_batch
        self.cells_per_group = cells_per_group
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            groups = rng.choice(len(self.pools), self.groups_per_batch, replace=False)
            yield np.concatenate([rng.choice(self.pools[g], self.cells_per_group, replace=True)
                                  for g in groups]).tolist()


def fitting_group_changes(adata, train_indices, layer: str, control: str) -> dict:
    """Use obs.celltype as the pairing context and all supplied fitting cells."""
    indices = np.asarray(train_indices, dtype=np.int64)
    grouped = adata.obs.iloc[indices].groupby(['celltype', 'genotype'], observed=True, sort=True).indices
    means, counts = {}, {}
    row_groups = np.full(adata.n_obs, -1, dtype=np.int64)
    for group, (key, positions) in enumerate(grouped.items()):
        rows = indices[positions]
        x = adata.layers[layer][rows]
        x = x.toarray() if sparse.issparse(x) else np.asarray(x)
        means[key] = x.astype(np.float32).mean(0, dtype=np.float64)
        counts[key] = len(rows)
        row_groups[rows] = group
    changes, weights = [], []
    for context, perturbation in grouped:
        changes.append(means[(context, perturbation)] - means[(context, control)])
        weights.append(1 / (1 / counts[(context, control)] + 1 / counts[(context, perturbation)]))
    return dict(row_groups=torch.as_tensor(row_groups), changes=torch.tensor(np.asarray(changes), dtype=torch.float32),
                weights=torch.tensor(weights, dtype=torch.float32))
