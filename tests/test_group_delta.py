"""Native group-delta integration: fitting scope, scale, sampling and checkpoint retention."""
import copy
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from scipy import sparse

from perttf.custom_loss import group_change_mse
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.expr_sampler import sample_source_counts
from perttf.utils.group_delta import ContextKOBatchSampler, fitting_group_changes


def inputs():
    counts = np.array([[1,2,0],[2,1,1],[1,0,3],[3,1,2], [1,2,1],[2,1,0],
                       [2,3,1],[3,1,2],[2,1,4],[4,2,1], [3,3,1],[2,2,4]], dtype=np.float32)
    data = ad.AnnData(sparse.csr_matrix(np.log1p(counts)),
        obs=pd.DataFrame({'celltype':['C']*12, 'genotype':['WT']*6+['KO']*6, 'sf':[1.]*12},
                         index=[f'c{i}' for i in range(12)]),
        var=pd.DataFrame({'highly_variable':[True]*3}, index=['a','b','c']))
    data.layers['X_binned'] = data.X.copy()
    return data, np.array([0,1,2,3,6,7,8,9]), np.array([4,5,10,11])


def model_and_config(next_decoder='group_delta'):
    data, _, _ = inputs()
    config = dict(layer_size=16, d_hid=16, nhead=4, nlayers=1, dropout=0.,
        GEPC=True, distribution='nb', n_ps=0, pert_dim=16, pert_pad_id=0,
        pert_sources={'fixture':(['KO'],torch.tensor([[1.,2.,3.]]))},
        use_fast_transformer=False, next_decoder=next_decoder, delta_rank=4,
        next_cell_pred_type='pert', next_weight=10., this_weight=1.,
        cell_type_classifier=False, genotype_classifier=False, CCE=False, ps_weight=0.,
        perturbation_input=False, batch_size=4, groups_per_batch=2, cells_per_group=2,
        seed=42, mask_ratio=.5, n_hvg=2, non_hvg_size=0, sampling_mode='hvg',
        size_factor_col='sf', amp=False, epochs=1, do_train=True,
        optimizer='muon', lr=.005, muon_aux_lr_ratio=.01, weight_decay=0.,
        scheduler='cosine', warmup_epochs=0., perturbation_metric_mode='mean',
        perturbation_checkpoint_metric='pearson_delta_mean', perturbation_metric_min_cells=2,
        perturbation_metric_real_scale='log1p',
        pairing_config=dict(pairing_anchor='target',source_selection='strict',target_selection='all',
                            target_sampling='cell',pair_schedule='dynamic',identity_condition='source',control_value='WT'))
    torch.manual_seed(42)
    return HFPerturbationTFModel.from_adata(data, config), config


def test_fitting_only_targets_and_group_sampler():
    data, fit, valid = inputs()
    targets = fitting_group_changes(data, fit, 'X_binned', 'WT')
    altered = data.copy();x = altered.layers['X_binned'].toarray();x[valid] = 100
    altered.layers['X_binned'] = sparse.csr_matrix(x)
    again = fitting_group_changes(altered, fit, 'X_binned', 'WT')
    for key in targets:torch.testing.assert_close(targets[key], again[key], rtol=0, atol=0)
    assert (targets['row_groups'][valid] == -1).all()
    assert torch.equal(targets['changes'][1], torch.zeros(3))  # sorted KO, WT
    sampler = ContextKOBatchSampler(targets['row_groups'][fit].numpy(), 3, 2, 2, 42)
    sampler.epoch = 1
    first = list(sampler);assert first == list(sampler)
    for batch in first:
        _, sizes = np.unique(targets['row_groups'][fit][batch], return_counts=True)
        np.testing.assert_array_equal(sizes, [2,2])


def test_effective_change_loss_keeps_wt_weight():
    source = torch.tensor([[1.,0.],[3.,1.],[2.,4.]])
    raw_delta = torch.tensor([[-5.,2.],[-1.,2.],[0.,0.]], requires_grad=True)
    prediction = (source+raw_delta).clamp_min(0)
    loss = group_change_mse(prediction, source, torch.tensor([0,0,1]),
        torch.tensor([[-2.,1.],[0.,0.]]), torch.tensor([2.,6.]))
    torch.testing.assert_close(loss, torch.tensor(.25))
    loss.backward()
    assert raw_delta.grad[0,0] == 0  # clipped raw change is not the supervised change


def test_native_hf_mean_sample_and_wt_identity(tmp_path):
    model, _ = model_and_config()
    with torch.no_grad():model.delta_decoder.shared.weight.normal_(0,.03)
    data, _, _ = inputs();sources=data[:4].copy();sources.obs['genotype_next']=['WT','KO','KO','WT']
    mean=model.predict_perturbations(sources,prediction_mode='mean',device='cpu')
    sample=model.predict_perturbations(sources,prediction_mode='sample',prediction_seed=9,device='cpu')
    expected=sample_source_counts(sources.X.toarray(),mean.obsm['mvc_next_expr'],np.ones((4,1)),np.random.default_rng(9))
    np.testing.assert_array_equal(sample.obsm['mvc_next_expr'],expected)
    np.testing.assert_array_equal(mean.obsm['mvc_next_expr'][[0,3]],sources.X.toarray()[[0,3]])
    np.testing.assert_array_equal(sample.obsm['mvc_next_expr'][[0,3]],sources.X.toarray()[[0,3]])
    assert mean.uns['perttf_prediction']['native_scale']=='log1p'
    assert mean.obsm['mvc_next_expr'].shape==(4,3)
    model.save_pretrained(tmp_path/'model')
    reloaded=HFPerturbationTFModel.from_pretrained(tmp_path/'model',strict=True)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,reloaded.state_dict()[name],rtol=0,atol=0)
    restored=reloaded.predict_perturbations(sources,prediction_mode='mean',device='cpu')
    np.testing.assert_array_equal(restored.obsm['mvc_next_expr'],mean.obsm['mvc_next_expr'])


def test_adapt_legacy_checkpoint_without_resetting_source_model(tmp_path):
    model, _ = model_and_config(next_decoder='mvc')
    model.save_pretrained(tmp_path/'legacy')
    adapted=HFPerturbationTFModel.from_pretrained(tmp_path/'legacy',strict=True)
    original={k:v.clone() for k,v in adapted.state_dict().items() if not k.startswith('pert_exp_encoder.')}
    adapted.enable_group_delta(rank=4)
    for key,value in original.items():torch.testing.assert_close(adapted.state_dict()[key],value,rtol=0,atol=0)
    torch.testing.assert_close(adapted.delta_decoder.W.weight,adapted.mvc_decoder.W.weight,rtol=0,atol=0)
    assert adapted.delta_decoder.W.weight.data_ptr()!=adapted.mvc_decoder.W.weight.data_ptr()
    assert adapted.delta_decoder.shared.weight.count_nonzero()==0
    with pytest.raises(ValueError,match='already exists'):adapted.enable_group_delta(rank=4)


def test_count_sampler_moments_and_depth():
    n=50000;sf=np.tile([[.25],[2.]],(n//2,1))
    source=np.log1p(np.tile([2,0,5],(n,1))*sf).astype(np.float32)
    mean=np.log1p(np.tile([1,3,5],(n,1))*sf).astype(np.float32)
    sample=sample_source_counts(source,mean,sf,np.random.default_rng(42))
    counts=np.expm1(sample.astype(np.float64))/sf
    np.testing.assert_allclose(counts.mean(0),[1,3,5],atol=.03)
    np.testing.assert_allclose((sample[:,:2]==0).mean(0),[.25,np.exp(-3)],atol=.006)
    np.testing.assert_array_equal(sample[:,2],source[:,2])
    with pytest.raises(ValueError,match='integer counts'):
        sample_source_counts(np.array([[.25,1.]]),np.ones((1,2)),np.ones((1,1)),np.random.default_rng(1))


def test_native_training_retains_final_before_restoring_best(tmp_path, monkeypatch):
    from perttf.model import train_function
    from perttf.model.config_gen import generate_config
    from perttf.model.train_data_gen import produce_training_datasets
    model, settings = model_and_config();data,fit,valid=inputs()
    complete=OmegaConf.to_container(model._init_default_train_config_(),resolve=True)
    complete.update(copy.deepcopy(settings))
    config, run=generate_config(complete,wandb_mode='disabled')
    generated=produce_training_datasets(data,config,vocab=model.vocab,
        genotype_to_index=model.genotype_to_index,cell_type_to_index=model.cell_type_to_index,
        train_indices=fit,valid_indices=valid)
    expected=data.layers['X_binned'][[4,5]].toarray().mean(0)
    np.testing.assert_allclose(generated['perturbation_reference_groups'][('C','WT')].mean,expected)
    run.finish()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    scores=iter([1.,0.])
    monkeypatch.setattr(train_function,'compute_perturbation_metrics',
        lambda *a,**kw:dict(aggregate={'pearson_delta':next(scores)},per_group={}))
    model.run_train(data,train_config=settings,train_indices=fit,valid_indices=valid,
                    save_dir=str(tmp_path/'trained'),device=torch.device('cpu'))
    assert model.checkpoint_selection['selected_epoch']==0
    for name,value in initial.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    final=HFPerturbationTFModel.from_pretrained(tmp_path/'trained/hf_final_model',strict=True)
    assert any(not torch.equal(initial[k],v) for k,v in final.state_dict().items())
    saved=json.loads((tmp_path/'trained/training_config.json').read_text())
    assert saved['checkpoint_selection']['final_epoch']==1
    assert saved['checkpoint_selection']['selected_epoch']==0
