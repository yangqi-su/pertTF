# Observed-WT group-delta prediction

This optional mode integrates the tested group-change model and source-preserving
count sampler into native `HFPerturbationTFModel`. Source MLM/MVC reconstruction
and KO prediction have separate expression semantics: the source MVC uses NB,
while the KO head returns **stored log1p expression**.

## Model and training objective

```
predicted_KO = max(observed_WT_log1p + shared_KO_effect + context_adjustment, 0)
```

The shared term depends on the perturbation embedding. A rank-64 interaction of
the source cell embedding and perturbation embedding supplies the adjustment.
The head's gene decoder is copied from source MVC; its response outputs start at
zero. Unified perturbation features give the control an exactly zero embedding,
so WT predictions exactly copy their observed source.

The supported recipe uses batch-label-free NB source reconstruction with an
inner-product MVC decoder and unified perturbation features (e.g. frozen ESM2).
The next-cell representation is an expression change, not a learned shifted
latent embedding; the reported `X_scGPT_next` remains the source embedding.

`obs.celltype` defines the context for pairing and group targets; `obs.genotype`
defines the perturbation. Set these columns explicitly for the scientific task.
Means and reliability counts use **only train_indices**, including fitting WT.
Validation cells never determine fitting targets. Input layers must contain real
log1p-normalized values, even if a legacy layer is named `X_binned`.

The group objective is the gene-average squared error between:

- the batch-group mean of `predicted_KO - actual_WT_source`, after zero clipping;
- the fitting KO-group mean minus the fitting WT-group mean.

Each group has weight `1 / (1/n_fit_WT + 1/n_fit_KO)`. WT identity groups have
zero change error but retain their weight in the denominator. Batches contain
eight distinct context/target groups, drawn uniformly, with sixteen target rows
per group sampled with replacement. Native pairing selects WT sources. The
number of updates follows the original fitting-row batch budget.

Recommended objective: source masked MLM-MSE + source masked MVC-NB +
`10 * group_change_mse`. Disable cell-type/genotype classifiers, CCE and PS losses
for this recipe. The native `next_weight` controls group loss weight in this mode.

## Construct or adapt

```python
from perttf.model.hf import HFPerturbationTFModel

# Adapt an existing unified-feature NB checkpoint; construct a fresh optimizer later.
model = HFPerturbationTFModel.from_pretrained(checkpoint, strict=True)
model.enable_group_delta(rank=64)

# Or create fresh weights through the regular constructor/from_adata:
# config includes next_decoder='group_delta', delta_rank=64, distribution='nb',
# GEPC=True, pert_sources=..., and the usual architecture settings.
```

`enable_group_delta` copies the currently loaded MVC gene decoder and preserves
the source model weights. It refuses to reset an existing group-delta head.
Architecture settings are stored in HF config; native strict reload reconstructs
the head without an experiment-specific subclass.

See [`demos/train_group_delta.py`](../demos/train_group_delta.py) for the full
adaptation recipe with explicit memberships. Muon/cosine uses peak matrix LR
.005, auxiliary Adam .00005, zero decay and one warmup epoch over50epochs.
This uses the existing native optimizer implementation.

Training uses ordinary `model.run_train(...)` / `wrapper_train(...)`. In this
mode validation starts at epoch0 and compares **unaltered stored log1p** group
moments. Use `perturbation_metric_mode='mean'` and
`perturbation_checkpoint_metric='pearson_delta_mean'`. Strict improvement selects
the earliest best checkpoint. With `run_train(save_dir=...)`:

- `save_dir/` is the validation-best HF model restored into the returned object.
- `save_dir/hf_final_model/` retains the final HF weights before best restoration.
- `training_config.json` records selected/final epochs under `checkpoint_selection`.
- `save_dir/wrapper_train/` contains native best/final state dictionaries.

## Mean and count-sampled predictions

Sources must be observed WT cells, with one requested target per row in
`obs.genotype_next`. The supplied rows and gene order are preserved.

```python
mean = model.predict_perturbations(
    sources, input_layer='X_binned', prediction_mode='mean', device='cuda')

draws = [model.predict_perturbations(
    sources, input_layer='X_binned', prediction_mode='sample',
    prediction_seed=seed, device='cuda') for seed in range(5)]

# Each result.obsm['mvc_next_expr'] is already log1p expression.
# Evaluate draws separately; averaging them is not population generation.
```

The sampler recovers source counts as `round(expm1(source_log)/sf)` and desired
counts as `expm1(predicted_log)/sf`. Decreases use binomial thinning; increases
keep source counts and add Poisson counts. Output is `log1p(sampled_counts*sf)`.
There is no new row-total normalization. Exactly unchanged predictions preserve
their source values. Expected counts are preserved; expected log-expression can
shift because log1p is nonlinear.

Here sf uses pertTF's convention: normalized counts = raw counts * sf. Prefer
an independently recorded factor through the existing `size_factor_col` setting.
Otherwise `_get_sf` assumes a count-one feature remains in the source panel;
near-integer recovery alone cannot prove that assumption. Sampling checks positive
factors and near-integer source recovery. The inferred/generated KO scale is
recorded as log1p in `uns['perttf_prediction']`, regardless of source MVC's NB scale.

One seed produces one draw, with a NumPy generator advanced in loader-batch order.
For exact replay of an external query-level seed scheme, request native means
and call `perttf.model.expr_sampler.sample_source_counts` once per frozen query
with its specified factors and RNG. Source sf should be computed from the same
gene panel or supplied from the original normalization metadata.

The frozen scientific splits, benchmark source/reference memberships, and shared
`vcb` evaluation remain experiment responsibilities. No benchmark data or Slurm
policy is embedded in the package.
