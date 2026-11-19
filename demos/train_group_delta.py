"""Fine-tune a unified-feature NB checkpoint with group-change supervision.

Run real training on a compute node. AnnData must already contain stored log1p
expression in the chosen layer, obs.celltype/genotype, and var.highly_variable.
The membership CSV has cell_id, fit, validation columns with boolean values.
The caller supplies the scientific split; this example never invents a split.
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from perttf.model.hf import HFPerturbationTFModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--adata',required=True)
    parser.add_argument('--membership',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-layer',default='X_binned')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    adata=ad.read_h5ad(args.adata)
    membership=pd.read_csv(args.membership).set_index('cell_id').loc[adata.obs_names]
    torch.manual_seed(42)
    model=HFPerturbationTFModel.from_pretrained(args.checkpoint,strict=True)
    model.enable_group_delta(rank=64)
    settings=dict(seed=42,epochs=50,batch_size=128,groups_per_batch=8,cells_per_group=16,
        next_cell_pred_type='pert',next_weight=10.,this_weight=1.,
        cell_type_classifier=False,genotype_classifier=False,CCE=False,ADV=False,
        perturbation_input=False,ps_weight=0.,mvc_masked_train=True,
        optimizer='muon',lr=.005,muon_aux_lr_ratio=.01,weight_decay=0.,
        scheduler='cosine',warmup_epochs=1.,
        sampling_mode='hvg',n_hvg=2000,non_hvg_size=1000,mask_ratio=.15,
        perturbation_metric_mode='mean',perturbation_checkpoint_metric='pearson_delta_mean',
        pairing_config=dict(pairing_anchor='target',source_selection='strict',target_selection='all',
            target_sampling='cell',pair_schedule='dynamic',identity_condition='source',
            control_value=model.pert_encoder.control_label))
    model.run_train(adata,train_config=settings,
        train_indices=np.flatnonzero(membership.fit.to_numpy()),
        valid_indices=np.flatnonzero(membership.validation.to_numpy()),
        input_layer_key=args.input_layer,save_dir=str(args.output),device=torch.device('cuda'))


if __name__=='__main__':
    main()
