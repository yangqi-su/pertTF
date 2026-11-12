"""Nuisance-label pairing: pair tables, rule lookup, nuisance shifts, energy score and a training run."""

import pytest

pytest.importorskip("scorpus")

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import torch
import wandb
from scipy import sparse

from scorpus import concat, from_h5ad
from scorpus.pairing import Sampler, make_groups
from perttf.custom_loss import energy_score_loss, group_energy_distance, straight_through_counts
from perttf.model.corpus_data import next_group_means, produce_corpus_datasets
from perttf.model.corpus_pairs import (PairRules, PerturbationBlockSampler, branch_weights, development_pairs,
                                       same_celltype_pairs)
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.train_function import evaluate, wrapper_train
from perttf.utils.custom_tokenizer import SimpleVocab

PAIR_COLUMNS = dict(perturbation="genotype", background="background", pool="pool", day="day", control="WT",
                    reference_background="H1", reference_pool="ko")


def toy_groups():
    rng = np.random.default_rng(0)
    rows = []
    for genotype, background, pool in [("WT", "H1", "ko"), ("WT", "H1-111", "ko"), ("A", "H1", "ko"), ("A", "HUES8", "ko"),
                                       ("B", "HUES8", "ko"), ("WT", "H1-111", "ext"), ("WT", "HUES8", "ext")]:
        for day, celltype in [("D-1", "ESC"), ("D3", "DE"), ("D7", "PFG"), ("D7", "Liver")]:
            rows += [(genotype, background, pool, day, celltype)] * int(rng.integers(3, 8))
    meta = pl.DataFrame(rows, schema=["genotype", "background", "pool", "day", "celltype"], orient="row")
    rank = {"D-1": 0, "D3": 1, "D7": 2}
    return make_groups(meta, by=["genotype", "background", "pool", "celltype", "day"]).with_columns(
        day_rank=pl.col("day").replace_strict(rank))


def test_same_celltype_pairs_and_weights():
    groups = toy_groups()
    pairs = same_celltype_pairs(groups, celltype="celltype", exclude_target_celltypes=("ESC",), **PAIR_COLUMNS)
    by_rule = {r: t for (r,), t in pairs.partition_by("rule", as_dict=True).items()}
    assert (by_rule["P1"]["source_genotype"] == "WT").all() and (by_rule["P1"]["target_background"] == "H1").all()
    assert (by_rule["P2"]["target_background"] == "HUES8").all()
    for rule in ("P1", "P2", "P3", "P6"):
        assert (by_rule[rule]["source_celltype"] == by_rule[rule]["target_celltype"]).all()
    assert not (pairs.filter(pl.col("rule").is_in(["P1", "P2"]))["target_celltype"] == "ESC").any()
    assert (by_rule["P3"]["source_background"] != by_rule["P3"]["target_background"]).all()
    assert (by_rule["P6"]["source_pool"] != by_rule["P6"]["target_pool"]).all()
    assert (by_rule["P5"]["source_group"] == by_rule["P5"]["target_group"]).all() and (by_rule["P5"]["link"] == "self").all()
    shares = {"P1": 0.6, "P2": 0.1, "P3": 0.1, "P5": 0.2, "P6": 0.0}
    weighted = branch_weights(pairs, shares, {"P5": []})
    totals = dict(weighted.group_by("rule").agg(pl.col("w").sum()).iter_rows())
    assert set(totals) == {"P1", "P2", "P3", "P5"}
    for rule, total in totals.items():
        assert total == pytest.approx(shares[rule])


def test_development_pairs_weights_follow_composition():
    groups = toy_groups()
    pairs = development_pairs(groups, day_rank="day_rank", origin_rank=0, **PAIR_COLUMNS)
    by_rule = {r: t for (r,), t in pairs.partition_by("rule", as_dict=True).items()}
    assert (by_rule["D1"]["source_day"] == "D-1").all() and (by_rule["D1"]["target_day"] != "D-1").all()
    assert (by_rule["D3"]["source_genotype"] == by_rule["D3"]["target_genotype"]).all()
    assert (by_rule["D3"]["target_day_rank"] > by_rule["D3"]["source_day_rank"]).all()
    assert (by_rule["D3"]["target_background"] == "H1").all()
    assert (by_rule["D5"]["source_pool"] == "ext").all()
    population = ["genotype", "background", "pool", "day"]
    weighted = branch_weights(pairs, {r: 1.0 for r in by_rule}, {"D1": population})
    split = weighted.filter((pl.col("rule") == "D1") & (pl.col("target_genotype") == "A")
                            & (pl.col("target_background") == "H1") & (pl.col("target_day") == "D7"))
    per_type = split.group_by("target_celltype", "target_n_cells").agg(pl.col("w").sum())
    ratio = per_type["w"] / per_type["target_n_cells"]
    assert ratio.max() == pytest.approx(ratio.min())


def test_pair_rules_match_sampled_pairs():
    groups = toy_groups()
    pairs = branch_weights(development_pairs(groups, day_rank="day_rank", origin_rank=0, **PAIR_COLUMNS),
                           {"D1": 0.4, "D3": 0.3, "D4": 0.1, "D5": 0.05, "D6": 0.05, "D8": 0.05, "D9": 0.05})
    sampler = Sampler(groups, pairs, links="link", weight="w")
    rules = PairRules(groups, pairs)
    samples = sampler.sample_batch(0, 4000, seed=1)
    source = np.array([s[0][0][0] for s in samples])
    target = np.array([s[0][1][0] for s in samples])
    codes = rules(source, target)
    share = np.bincount(codes, minlength=len(rules.names)) / len(codes)
    assert share[rules.names.index("D1")] == pytest.approx(0.4, abs=0.03)
    self_rows = codes == rules.names.index("D9")
    np.testing.assert_array_equal(source[self_rows], target[self_rows])
    # Reversed, a D1 pair goes from a later KO cell back to an origin WT cell, which no rule allows.
    d1 = np.flatnonzero(codes == rules.names.index("D1"))[:1]
    with pytest.raises(ValueError, match="not in the pair table"):
        rules(target[d1], source[d1])


def test_block_sampler_limits_perturbations_per_batch():
    groups = toy_groups()
    pairs = branch_weights(same_celltype_pairs(groups, celltype="celltype", **PAIR_COLUMNS),
                           {"P1": 0.55, "P2": 0.15, "P3": 0.1, "P5": 0.15, "P6": 0.05}, {"P5": []})
    sampler = PerturbationBlockSampler(groups, pairs, perturbation="genotype", control="WT", max_perturbations=1)
    rules = PairRules(groups, pairs)
    genotype = groups.table["genotype"].to_numpy()
    group_of_row = rules.group_of_row
    for step in range(5):
        samples = sampler.sample_batch(step, 64, seed=2)
        source = np.array([s[0][0][0] for s in samples])
        target = np.array([s[0][1][0] for s in samples])
        rules(source, target)  # every pair is in the pair table
        assert len(samples) == 64
        assert len(set(genotype[group_of_row[target]]) - {"WT"}) == 1


def test_energy_score_prefers_the_target_distribution():
    target = torch.tensor([[-1.0], [1.0]] * 50)
    mask = torch.ones_like(target, dtype=torch.bool)
    mean = torch.zeros_like(target)
    signs = torch.tensor([[-1.0], [1.0]] * 50)
    bimodal_a, bimodal_b = signs, -signs  # two independent-looking draws covering both modes
    assert energy_score_loss(bimodal_a, bimodal_b, target, mask) < energy_score_loss(mean, mean, target, mask)


def small_model(spec, noise_dim=0, next_delta=False, next_delta_relative=False):
    vocab = SimpleVocab.from_dict({"<pad>": 0, "<cls>": 1, "g1": 2, "g2": 3, "g3": 4})
    torch.manual_seed(0)
    return HFPerturbationTFModel(
        n_pert=3, nlayers_pert=2, n_ps=0, d_model=16, nhead=2, d_hid=32, nlayers=1, vocab=vocab, dropout=0.0,
        do_mvc=True, use_batch_labels=False, n_bins=0, ecs_threshold=0.0, n_cls=2, input_emb_style="continuous",
        genotype_to_index={"WT": 0, "A": 1, "B": 2}, cell_type_to_index={"x": 0, "y": 1}, control_label="WT",
        nuisance_spec=spec, noise_dim=noise_dim, next_delta=next_delta, next_delta_relative=next_delta_relative,
    ).eval()


@pytest.mark.parametrize("context, mode", [("source", "additive"), ("wt_counterfactual", "both"), ("state", "inject")])
def test_no_labels_predict_the_reference(context, mode):
    spec = {"levels": {"background": 3}, "reference": {"background": 0}, "context": {"background": context},
            "mode": {"background": mode}}
    model = small_model(spec)
    assert model._hub_mixin_config["nuisance_spec"].startswith("{")
    src = torch.tensor([[1, 2, 3, 4]] * 4)
    values = torch.rand(4, 4)
    kwargs = dict(pert_labels_next=torch.tensor([1, 2, 0, 1]), MVC=True)
    reference = {"background": torch.zeros(4, dtype=torch.long)}
    other = {"background": torch.full((4,), 2)}
    with torch.no_grad():
        none = model(src, values, src.eq(0), **kwargs)["mvc_output_next"]["pred"]
        ref = model(src, values, src.eq(0), nuisance_labels=reference, nuisance_labels_next=reference, **kwargs)["mvc_output_next"]["pred"]
        shifted = model(src, values, src.eq(0), nuisance_labels=reference, nuisance_labels_next=other, **kwargs)["mvc_output_next"]["pred"]
    torch.testing.assert_close(none, ref)
    assert not torch.allclose(ref, shifted)


def test_pert_shift_excludes_target_nuisance_shifts():
    spec = {"levels": {"background": 3}, "reference": {"background": 0}, "context": {"background": "source"}}
    model = small_model(spec)
    src = torch.tensor([[1, 2, 3, 4]] * 4)
    values = torch.rand(4, 4)
    kwargs = dict(pert_labels_next=torch.tensor([1, 2, 0, 1]), MVC=True, pert_labels_control=torch.zeros(4, dtype=torch.long),
                  nuisance_labels={"background": torch.zeros(4, dtype=torch.long)})
    with torch.no_grad():
        ref = model(src, values, src.eq(0), nuisance_labels_next={"background": torch.zeros(4, dtype=torch.long)}, **kwargs)
        other = model(src, values, src.eq(0), nuisance_labels_next={"background": torch.full((4,), 2)}, **kwargs)
    torch.testing.assert_close(ref["pert_shift"], other["pert_shift"])
    torch.testing.assert_close(ref["pert_shift"][2], torch.zeros_like(ref["pert_shift"][2]))  # the control's own shift is zero
    assert not torch.allclose(ref["contrastive_dict"]["next_emb0"], other["contrastive_dict"]["next_emb0"])


def test_next_delta_starts_as_the_source_cell():
    model = small_model(None, next_delta=True)
    src = torch.tensor([[1, 2, 3, 4]] * 3)
    values = torch.rand(3, 4) * 3
    with torch.no_grad():
        out = model(src, values, src.eq(0), pert_labels_next=torch.tensor([0, 1, 2]), MVC=True, mvc_src=src,
                    mvc_src_values=values)
    expected = torch.log1p(torch.expm1(values) + np.exp(-3.0))
    torch.testing.assert_close(out["mvc_output_next"]["pred"], expected)
    assert out["mvc_output"]["pred"].shape == expected.shape   # the MVC decoder still decodes the source


def test_relative_next_delta_leaves_control_cells_unchanged():
    model = small_model(None, noise_dim=4, next_delta=True, next_delta_relative=True)
    torch.nn.init.normal_(model.next_decoder.W_delta.weight)   # a trained-looking head
    torch.nn.init.normal_(model.next_decoder.W_add.weight)
    src = torch.tensor([[1, 2, 3, 4]] * 3)
    values = torch.rand(3, 4) * 3
    with torch.no_grad():
        out = model(src, values, src.eq(0), pert_labels_next=torch.tensor([0, 1, 2]), MVC=True, mvc_src=src,
                    mvc_src_values=values, n_noise_draws=2)
    for draw in out["mvc_output_next_draws"]:
        torch.testing.assert_close(draw["pred"][0], values[0])   # WT (ID 0): the source itself, for any noise
    assert not torch.allclose(out["mvc_output_next"]["pred"][1], values[1])


def test_straight_through_counts_are_counts_around_the_prediction():
    torch.manual_seed(0)
    sf = torch.full((20000, 1), 2.0)                      # normalised = counts x 2
    source = torch.log1p(torch.tensor([[2.0, 0.0, 6.0]]) * 2).expand(20000, -1)   # raw counts 2, 0, 6
    pred = torch.log1p(torch.tensor([[1.0, 0.05, 6.0]]) * 2).expand(20000, -1).clone().requires_grad_(True)
    drawn = straight_through_counts(pred, source, sf)
    counts = torch.expm1(drawn.detach()) / sf
    assert torch.allclose(counts, counts.round(), atol=1e-3)
    assert torch.allclose(counts.mean(0), torch.tensor([1.0, 0.05, 6.0]), atol=0.03)
    assert abs((counts[:, 0] == 0).float().mean().item() - 0.25) < 0.02   # Binomial(2, 1/2)
    assert torch.allclose(counts[:, 2], torch.full((20000,), 6.0), atol=1e-4)   # unchanged gene, unchanged counts
    drawn.sum().backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad > 0).all()


def test_group_energy_distance_compares_distributions():
    torch.manual_seed(0)
    projection = torch.eye(4)
    same = torch.randn(400, 4), torch.randn(400, 4)
    shifted_spread = torch.randn(400, 4), torch.randn(400, 4) * 3
    groups = torch.zeros(400, dtype=torch.long)
    assert abs(group_energy_distance(*same, groups, projection).item()) < 0.05
    assert group_energy_distance(*shifted_spread, groups, projection).item() > 0.5   # same mean, different spread


def test_next_group_means_match_full_expr(tmp_path):
    corpus, obs = nuisance_corpus(tmp_path)
    rows = np.arange(len(obs))
    with wandb.init(mode="disabled", config=dict(next_cell_pred_type="pert", GEPC=True, batch_size=8, max_seq_len=4,
                                                 sampling_mode="simple", seed=0)) as run:
        data_gen = produce_corpus_datasets(
            corpus, run.config, train_indices=rows[(obs["genotype"] == "ctrl") | (rows % 2 == 0)],
            valid_indices=rows[(obs["genotype"] == "ctrl") | (rows % 2 == 1)],
            perturbation_column="genotype", celltype_column="celltype", batch_column="batch", control_label="ctrl",
            tokenizer_stoi={"<pad>": 0, "<cls>": 1, "g1": 2, "g2": 3, "g3": 4},
        )
    batch = next(iter(data_gen["train_loader"]))
    # One group per cell: each group mean is that cell's own full_expr.
    cells = batch["index"].numpy()
    own = next_group_means(corpus, cells, cells[:, None], [len(obs)], 10000.0)
    torch.testing.assert_close(own["means"][own["table"][cells]][:, 1:], batch["full_expr"][:, 1:])


def nuisance_corpus(tmp_path):
    rng = np.random.default_rng(3)
    cells = []
    for genotype, background, pool in [("ctrl", "H1", "ko"), ("ctrl", "H1-111", "ko"), ("P1", "H1", "ko"),
                                       ("P1", "HUES8", "ko"), ("ctrl", "H1-111", "ext")]:
        for celltype in ("T", "U"):
            cells += [(genotype, background, pool, celltype, "D3")] * 4
    obs = pd.DataFrame(cells, columns=["genotype", "background", "pool", "celltype", "day"])
    obs["batch"] = "a"
    obs.index = [f"cell{i}" for i in range(len(obs))]
    counts = rng.integers(0, 6, size=(len(obs), 3)).astype(np.int32)
    data = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))
    data.write_h5ad(tmp_path / "a.h5ad")
    member = from_h5ad(tmp_path / "a.h5ad", tmp_path / "a", chunk_rows=8)
    columns = ["genotype", "background", "pool", "celltype", "day", "batch"]
    return concat({"a": member}, obs_columns={c: {"a": c} for c in columns}, feature_key={"a": "index"}), obs


@pytest.mark.parametrize("next_delta, group_loss, relative, group_energy", [
    (False, False, False, False), (True, False, False, False), (True, True, False, False), (True, True, True, False),
    (True, True, True, True)])
def test_step_mode_training_with_nuisance_labels(tmp_path, monkeypatch, next_delta, group_loss, relative, group_energy):
    torch.set_num_threads(2)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    parameters = dict(
        next_cell_pred_type="pert", GEPC=True, input_emb_style="continuous",
        batch_size=8, max_seq_len=4, mask_ratio=0.5, sampling_mode="simple",
        seed=4, ps_weight=0.0, pad_token="<pad>", pad_value=-2, mask_value=-1,
        amp=False, dab_weight=0.0, ADV=False, lr=1e-3, schedule_ratio=0.9,
        log_interval=100, use_batch_label=False, perturbation_input=False,
        next_weight=1.0, this_weight=1.0, ecs_thres=0.0, CCE=False,
        explicit_zero_prob=False, distribution=None, mvc_masked_train=False,
        cell_type_classifier=True, genotype_classifier=True,
        cell_type_classifier_weight=0.1, perturbation_classifier_weight=0.1,
        perturbation_metric_min_cells=1, perturbation_metric_mode="mean",
        perturbation_metric_target_sum=10000.0, epochs=1, do_train=True,
        dataset_name="synthetic_nuisance", energy_weight=1.0, next_loss_on_self_pairs="control",
        nuisance_adversary_weight=0.5, shift_contrastive_weight=0.5, shift_contrastive_center=True,
        next_group_loss_weight=1.0 if group_loss else 0.0,
        next_group_energy_weight=1.0 if group_energy else 0.0,
    )
    if group_energy:
        np.save(tmp_path / "projection.npy", np.eye(3, 2))   # the synthetic corpus has 3 genes
        parameters["next_group_energy_projection"] = str(tmp_path / "projection.npy")
    if group_loss:
        parameters.update(next_weight=0.0, energy_weight=0.0)
    with wandb.init(mode="disabled", config=parameters) as run:
        corpus, obs = nuisance_corpus(tmp_path)
        is_clean = (obs["background"] == "H1") & (obs["pool"] == "ko")
        rank = np.arange(len(obs)) % 2
        valid = np.flatnonzero(is_clean & ((obs["genotype"] == "ctrl") | (rank == 1)))
        train = np.flatnonzero(~obs.index.isin(obs.index[valid]) | (obs["genotype"] == "ctrl"))
        metadata = corpus.metadata_index.df[train]
        groups = make_groups(metadata, by=["genotype", "background", "pool", "celltype", "day"])
        pairs = same_celltype_pairs(groups, perturbation="genotype", celltype="celltype", background="background",
                                    pool="pool", day="day", control="ctrl", reference_background="H1", reference_pool="ko")
        pairs = branch_weights(pairs, {"P1": 0.5, "P2": 0.2, "P3": 0.1, "P5": 0.1, "P6": 0.1}, {"P5": []})
        tokenizer = {"<pad>": 0, "<cls>": 1, "g1": 2, "g2": 3, "g3": 4}
        data_gen = produce_corpus_datasets(
            corpus, run.config, train_indices=train, valid_indices=valid,
            perturbation_column="genotype", celltype_column="celltype", batch_column="batch", control_label="ctrl",
            tokenizer_stoi=tokenizer, extra_label_columns={"background": "background", "pool": "pool"},
            extra_label_to_index={"background": {"H1": 0, "H1-111": 1, "HUES8": 2}, "pool": {"ko": 0, "ext": 1}},
            train_pairs=(groups, pairs), steps_per_epoch=3,
            next_group_labels=("background", "pool") if group_loss else None,
        )
        loader = data_gen["train_loader"]
        assert len(loader) == 3
        batch = next(iter(loader))
        assert {"background_labels_next", "pool_labels_next", "pair_rule"} <= set(batch)
        assert data_gen["pair_rule_names"] == ["P1", "P2", "P3", "P5", "P6"]
        self_rows = batch["pair_rule"] == data_gen["pair_rule_names"].index("P5")
        assert torch.equal(batch["index"][self_rows], batch["next_index"][self_rows])
        spec = {"levels": {"background": 3, "pool": 2}, "reference": {"background": 0, "pool": 0},
                "context": {"background": "source", "pool": "state"},
                "conditions": {"celltype": len(data_gen["cell_type_to_index"])}}
        model = HFPerturbationTFModel(
            n_pert=len(data_gen["genotype_to_index"]), nlayers_pert=2, n_ps=0, d_model=16, nhead=2, d_hid=32, nlayers=1,
            vocab=data_gen["vocab"], dropout=0.0, pad_token="<pad>", pad_value=-2, do_mvc=True, do_dab=False,
            use_batch_labels=False, num_batch_labels=1, domain_spec_batchnorm=False, n_bins=0, ecs_threshold=0.0,
            explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False, n_cls=1, nlayers_cls=2,
            input_emb_style="continuous", genotype_to_index=data_gen["genotype_to_index"],
            cell_type_to_index=data_gen["cell_type_to_index"], control_label="ctrl",
            nuisance_spec=spec, noise_dim=4, next_delta=next_delta, next_delta_relative=relative,
        )
        before = model.nuisance_shift["background"].fc[0].weight.detach().clone()
        save_dir = tmp_path / "training"
        save_dir.mkdir()
        best = wrapper_train(model, run.config, data_gen, save_dir=save_dir, device=torch.device("cpu"))
        assert not torch.equal(before, model.nuisance_shift["background"].fc[0].weight)
        if group_loss:   # the per-cell next loss is off: only the group loss moves the delta head from zero
            assert model.next_decoder.W_delta.weight.abs().sum() > 0
        result = evaluate(
            best, data_gen["valid_loader"], run.config, data_gen["vocab"], device=torch.device("cpu"),
            cell_type_to_index=data_gen["cell_type_to_index"], genotype_to_index=data_gen["genotype_to_index"],
            expr_prediction_mode="mean",
        )
        assert all(np.isfinite(v) for v in result["losses"].values())
