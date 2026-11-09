"""Pair tables for training with nuisance labels (parent background, pool group), built on scorpus.pairing.

Groups must split cells by perturbation, cell type, background, pool and day (plus a numeric day
rank for development pairs). Each function returns a pair table with a ``rule`` and a ``link``
column for ``scorpus.pairing.Sampler(groups, pairs, links="link", weight="w")``; ``branch_weights``
adds ``w``.

Same-celltype pairing (source and target share a cell type):
  P1  WT -> KO, reference background target           (KO effect; the only rule scored by a clean truth)
  P2  WT -> KO, other-background target                (KO effect in e.g. HUES8 targets)
  P3  WT -> WT, background switch, same day            (background offset, e.g. clone 111)
  P5  every group with itself ("self")                 (reconstruction only)
  P6  WT -> WT, pool switch, same day and background   (pool offset)
Development pairing (target cell type is an outcome, not matched):
  D1  WT at the origin day -> KO at a later day         (KO applied at origin; the deployable query)
  D3  KO A -> KO A at a later day, reference background (the KO's own trajectory)
  D4  WT -> WT later day, reference pool                (development in the knockout pools)
  D5  WT -> WT later day, other pools                   (development in the external pools)
  D6  WT -> WT, background switch, same day
  D8  WT -> WT, pool switch, same day and background
  D9  every group with itself ("self")
KO-source background switches are left out: a KO source would carry its genotype into the
background branch's input.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scorpus.pairing import Groups, Sampler, combine, cross, src, tgt


def _rule(table: pl.DataFrame, name: str, link: str) -> pl.DataFrame:
    return table.with_columns(rule=pl.lit(name), link=pl.lit(link))


def same_celltype_pairs(
    groups: Groups, *, perturbation: str, celltype: str, background: str, pool: str, day: str,
    control: str, reference_background: str, reference_pool: str,
    source_backgrounds: list[str] | None = None, exclude_target_celltypes: tuple[str, ...] = (),
    rules: tuple[str, ...] = ("P1", "P2", "P3", "P5", "P6"),
) -> pl.DataFrame:
    """Pairs within one cell type. ``source_backgrounds``: backgrounds of WT sources for KO pairs
    (default: every background); ``exclude_target_celltypes``: cell types never used as KO targets (ESC)."""
    is_ctrl = pl.col(perturbation) == control
    in_ref_pool = pl.col(pool) == reference_pool
    wt_source = is_ctrl & in_ref_pool
    if source_backgrounds is not None:
        wt_source = wt_source & pl.col(background).is_in(source_backgrounds)
    ko_target = ~is_ctrl & in_ref_pool & ~pl.col(celltype).is_in(list(exclude_target_celltypes))
    branches = {
        "P1": lambda: _rule(cross(groups, on=[celltype, pool], source=wt_source,
                                  target=ko_target & (pl.col(background) == reference_background)), "P1", "random"),
        "P2": lambda: _rule(cross(groups, on=[celltype, pool], source=wt_source,
                                  target=ko_target & (pl.col(background) != reference_background)), "P2", "random"),
        "P3": lambda: _rule(cross(groups, on=[celltype, day, pool], source=is_ctrl & in_ref_pool, target=is_ctrl & in_ref_pool)
                            .filter(src[background] != tgt[background]), "P3", "random"),
        "P5": lambda: _rule(cross(groups, on=groups.by), "P5", "self"),
        "P6": lambda: _rule(cross(groups, on=[celltype, day, background], source=is_ctrl, target=is_ctrl)
                            .filter(src[pool] != tgt[pool]), "P6", "random"),
    }
    return combine(*(branches[name]() for name in rules))


def development_pairs(
    groups: Groups, *, perturbation: str, background: str, pool: str, day: str, day_rank: str,
    control: str, reference_background: str, reference_pool: str, origin_rank: int,
    source_backgrounds: list[str] | None = None,
    rules: tuple[str, ...] = ("D1", "D3", "D4", "D5", "D6", "D8", "D9"),
) -> pl.DataFrame:
    """Pairs across days; cell type is not matched. ``origin_rank``: day rank of the starting state (ESC)."""
    is_ctrl = pl.col(perturbation) == control
    in_ref_pool = pl.col(pool) == reference_pool
    origin_wt = is_ctrl & in_ref_pool & (pl.col(day_rank) == origin_rank)
    if source_backgrounds is not None:
        origin_wt = origin_wt & pl.col(background).is_in(source_backgrounds)
    ref_ko = ~is_ctrl & in_ref_pool & (pl.col(background) == reference_background)
    later = tgt[day_rank] > src[day_rank]
    branches = {
        "D1": lambda: _rule(cross(groups, on=[pool], source=origin_wt,
                                  target=~is_ctrl & in_ref_pool & (pl.col(day_rank) > origin_rank)), "D1", "random"),
        "D3": lambda: _rule(cross(groups, on=[perturbation, background, pool], source=ref_ko, target=ref_ko)
                            .filter(later), "D3", "random"),
        "D4": lambda: _rule(cross(groups, on=[background, pool], source=is_ctrl & in_ref_pool, target=is_ctrl & in_ref_pool)
                            .filter(later), "D4", "random"),
        "D5": lambda: _rule(cross(groups, on=[background, pool], source=is_ctrl & ~in_ref_pool, target=is_ctrl & ~in_ref_pool)
                            .filter(later), "D5", "random"),
        "D6": lambda: _rule(cross(groups, on=[day, pool], source=is_ctrl & in_ref_pool, target=is_ctrl & in_ref_pool)
                            .filter(src[background] != tgt[background]), "D6", "random"),
        "D8": lambda: _rule(cross(groups, on=[day, background], source=is_ctrl, target=is_ctrl)
                            .filter(src[pool] != tgt[pool]), "D8", "random"),
        "D9": lambda: _rule(cross(groups, on=groups.by), "D9", "self"),
    }
    return combine(*(branches[name]() for name in rules))


def branch_weights(pairs: pl.DataFrame, shares: dict[str, float],
                   populations: dict[str, list[str]] | None = None) -> pl.DataFrame:
    """Add ``w`` so that step-mode draws follow ``shares`` per rule.

    Within a rule, every target population gets the same weight; inside a population, target
    groups are weighted by their cells, and a target group's weight is split over its source
    groups by their cells. ``populations[rule]``: grouping columns that define a population (for
    development rules e.g. perturbation, background, pool, day, so that cell types split one
    population and their cell numbers reproduce its composition); default: each target group is
    its own population. ``[]`` makes the whole rule one population (weights follow cell numbers).
    Rules with share 0 are dropped.
    """
    populations = populations or {}
    weighted = []
    for rule, rows in pairs.partition_by("rule", as_dict=True, maintain_order=True).items():
        rule = rule[0]
        share = shares[rule]
        if share == 0:
            continue
        columns = populations.get(rule)
        pop = ["target_group"] if columns is None else [f"target_{c}" for c in columns]
        if not pop:
            rows, pop = rows.with_columns(__all=pl.lit(0)), ["__all"]
        targets = rows.unique("target_group").group_by(pop).agg(pop_cells=pl.col("target_n_cells").sum())
        rows = rows.join(targets, on=pop, how="left").with_columns(
            source_cells=pl.col("source_n_cells").sum().over("target_group"),
        )
        rows = rows.with_columns(w=(
            share / targets.height
            * pl.col("target_n_cells") / pl.col("pop_cells")
            * pl.col("source_n_cells") / pl.col("source_cells")
        )).drop("pop_cells", "source_cells", "__all", strict=False)
        weighted.append(rows)
    return pl.concat(weighted, how="diagonal_relaxed")


class PairRules:
    """The rule of each sampled (source row, target row), looked up from the pair table.

    The sampler returns row positions only; a pair table has one row per (source group, target
    group) and the sampler draws targets inside the pair row's target group, so the group pair
    identifies the rule exactly.
    """

    def __init__(self, groups: Groups, pairs: pl.DataFrame, column: str = "rule"):
        self.n_groups = groups.table.height
        self.group_of_row = np.empty(groups.n_rows, dtype=np.int64)
        for g, rows in enumerate(groups.rows):
            self.group_of_row[rows] = g
        self.names = sorted(pairs[column].unique().to_list())
        keys = pairs["source_group"].to_numpy() * self.n_groups + pairs["target_group"].to_numpy()
        order = np.argsort(keys)
        self.keys = keys[order]
        self.codes = pairs[column].replace_strict({n: i for i, n in enumerate(self.names)}, return_dtype=pl.Int64).to_numpy()[order]

    def __call__(self, source_rows: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
        query = self.group_of_row[source_rows] * self.n_groups + self.group_of_row[target_rows]
        position = np.minimum(np.searchsorted(self.keys, query), len(self.keys) - 1)
        # A pair from another sampler or row space would otherwise get a neighbouring rule silently.
        if not np.array_equal(self.keys[position], query):
            raise ValueError("Some sampled pairs are not in the pair table")
        return self.codes[position]


class PerturbationBlockSampler:
    """Step-mode sampler whose batches hold pairs of at most ``max_perturbations`` target perturbations.

    Pairs with a perturbed target (WT->KO and KO self pairs) are split by target perturbation. Each
    step draws ``max_perturbations`` perturbations without replacement, by their total pair weight,
    and the batch's perturbed-target pairs come only from them, split by weight. Pairs with a control
    target keep their overall weight share. Gives a contrastive loss over perturbations several
    positives per perturbation in every batch.
    """

    def __init__(self, groups: Groups, pairs: pl.DataFrame, *, perturbation: str, control: str,
                 max_perturbations: int, weight: str = "w"):
        target = pl.col(f"target_{perturbation}")
        control_pairs = pairs.filter(target == control)
        perturbed = pairs.filter(target != control)
        self.names = sorted(perturbed[f"target_{perturbation}"].unique().to_list())
        self.control_sampler = Sampler(groups, control_pairs, links="link", weight=weight)
        self.samplers = [Sampler(groups, perturbed.filter(target == name), links="link", weight=weight) for name in self.names]
        self.weights = np.array([perturbed.filter(target == name)[weight].sum() for name in self.names])
        self.control_share = control_pairs[weight].sum() / pairs[weight].sum()
        self.max_perturbations = max_perturbations

    def sample_batch(self, step: int, n: int, seed: int = 0) -> list:
        rng = np.random.default_rng([seed, step, 1])
        n_control = rng.binomial(n, self.control_share)
        chosen = rng.choice(len(self.names), size=min(self.max_perturbations, len(self.names)), replace=False,
                            p=self.weights / self.weights.sum())
        counts = rng.multinomial(n - n_control, self.weights[chosen] / self.weights[chosen].sum())
        # Each sub-sampler gets its own seed: with a shared one they would draw at the same random points.
        samples = self.control_sampler.sample_batch(step, n_control, seed=seed * 1009) if n_control else []
        for i, count in zip(chosen, counts):
            if count:
                samples += self.samplers[i].sample_batch(step, int(count), seed=seed * 1009 + int(i) + 1)
        return samples
