import json

from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from tqdm import trange
import numpy as np

import torch
from torch import nn
from torch.distributions import Bernoulli
import torch.nn.functional as F

import torch.distributed as dist

from .base_model import BaseModel
from .pert_encoder import KnockoutLookup, encode_perturbation_set, reduce_perturbation_embeddings

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .expr_sampler import DistributionGenerator
from .modules import (
    PertExpEncoder,
    PerturbationDecoder,
    PSDecoder,
    Batch2LabelEncoder,
    PertLabelEncoder,
    FeaturePertEncoder,
    NuisanceShift,
    NuisanceAdversary,
    NextDeltaDecoder,
    GroupDeltaDecoder,
    )


class PerturbationTFModel(BaseModel):
    def __init__(self,
                 n_pert: int,
                 nlayers_pert: int,
                 n_ps: int,
                 *args, **kwargs):
        self.pred_lochness_next = kwargs.pop("pred_lochness_next", False) # additional optional parameter to ask whether to predict lochness scores
        ps_decoder2_nlayer = kwargs.pop("ps_decoder2_nlayer",3) # additional parameter to specify ps_decoder2 nlayer
        self.pert_pad_id = kwargs.pop("pert_pad_id", None) # get the pert_pad_id
        self.pert_dim = kwargs.pop('pert_dim', None)
        self.pert_aggregation = kwargs.pop('pert_aggregation', 'sum')
        # optional perturbation FEATURE matrix (n_pert, feat_dim). When provided, the
        # perturbation encoder maps each genotype index through its feature vector + an
        # MLP instead of an independent lookup row, so novel (unseen-in-training)
        # conditions get a meaningful embedding -> enables zero-shot (regime-A) prediction.
        self.pert_features = kwargs.pop("pert_features", None)
        # Nuisance labels of source and target cells (e.g. parent background, pool group), as a dict
        # or its JSON string:
        #   levels:    {name: number of levels};  reference: {name: level with no shift}
        #   context:   {name: "source" | "state" | "wt_counterfactual"}, the detached state a target's
        #              shift is read from (default "source"); self-pairs always read their own embedding
        #   mode:      {name: "additive" | "inject" | "both"} (default "additive"): a shift added after
        #              the perturbation head (cannot depend on genotype), the label as an input of the
        #              perturbation head (may interact with genotype), or both, with the injected label
        #              replaced by a null level with probability inject_dropout during training
        #   conditions: {label: number of levels} given to the adversaries (e.g. celltype, day)
        #   adversary: whether each nuisance gets an adversary on the cell embedding (default True)
        #   control_index: perturbation ID of the control, for the WT counterfactual
        #   decoder_terms: with next_group_delta, the labels reach only the adversaries and the change head's line and
        #              pool terms (GroupDeltaDecoder batch_terms, line = "background"): no encoder input, shifts or injection
        spec = kwargs.pop("nuisance_spec", None)
        self.nuisance_spec = json.loads(spec) if isinstance(spec, str) else (spec or {})
        # Width of a Gaussian noise vector given to the perturbation head; 0 disables it.
        self.noise_dim = kwargs.pop("noise_dim", 0)
        # Decode next states with NextDeltaDecoder (a change applied to the source's observed expression,
        # forward's mvc_src_values) instead of the MVC decoder, which still decodes the source itself.
        self.next_delta = kwargs.pop("next_delta", False)
        # Perturbation ID of the control. With next_delta, the head then reads the next state minus the
        # control's state for the same cell and noise (with no bias), so the control leaves a cell unchanged.
        self.next_delta_control = kwargs.pop("next_delta_control", None)
        # With next_delta and next_delta_control: decode next cells with GroupDeltaDecoder instead, from the source
        # cell's embedding and the target's perturbation embedding (the perturbation head's state is not used).
        self.next_group_delta = kwargs.pop("next_group_delta", False)
        # With next_group_delta: decreases scale the source by the ratio that lowers the context's WT mean by D (see
        # GroupDeltaDecoder). The WT means are context_wt_mean, (cell types, mvc_src genes) normalised counts in the order
        # of the full gene list, set by the caller (not saved with the weights); forward then needs celltype_labels.
        self.next_group_delta_wt_rule = kwargs.pop("next_group_delta_wt_rule", False)
        # With next_group_delta: D is a change of mean log1p, added to each cell's log1p values (GroupDeltaDecoder shift_rule);
        # train with next_group_delta_scale "mean_of_log" so the group targets are on that scale.
        next_group_delta_shift = kwargs.pop("next_group_delta_shift", False)
        # With next_group_delta: one undecomposed network for D (GroupDeltaDecoder single), the decomposition's ablation.
        next_group_delta_single = kwargs.pop("next_group_delta_single", False)
        self.context_wt_mean = None
        # Perturbation IDs with a row in the knockout look-up (KnockoutLookup), and the chance that a knockout is hidden
        # from its own row in training. With a table, a target knockout is represented through the look-up (the control
        # keeps its own embedding); needs next_delta_control and single-perturbation labels.
        pert_lookup = kwargs.pop("pert_lookup", None)
        pert_lookup_hide = kwargs.pop("pert_lookup_hide", 0.3)
        pert_lookup_self_bonus = kwargs.pop("pert_lookup_self_bonus", False)
        pert_lookup_hard = kwargs.pop("pert_lookup_hard", False)
        # Perturbation IDs seen in training. Any other target knockout gets one learned "unknown knockout" embedding,
        # which training fits (next_unknown_weight) to each cell type's average knockout change, context_avg_change:
        # (cell types, mvc_src genes) changes of log1p of group means, NaN rows for cell types without one, set by the
        # caller (not saved with the weights). Needs next_delta_control.
        pert_seen = kwargs.pop("pert_seen", None)
        self.context_avg_change = None
        super().__init__(*args, **kwargs)
        # add perturbation encoder
        # variables are defined in super class
        d_model = self.d_model
        pert_dim = d_model if self.pert_dim is None else self.pert_dim
        #self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        if self.pert_features is not None:
            self.pert_encoder = FeaturePertEncoder(
                self.pert_features, pert_dim, padding_idx=self.pert_pad_id)
        else:
            self.pert_encoder = PertLabelEncoder(n_pert, pert_dim, padding_idx=self.pert_pad_id)
        levels = self.nuisance_spec.get("levels", {})
        reference = self.nuisance_spec.get("reference", {})
        modes = {name: self.nuisance_spec.get("mode", {}).get(name, "additive") for name in levels}
        decoder_terms = self.nuisance_spec.get("decoder_terms", False)
        model_levels = {} if decoder_terms else levels
        self.nuisance_input = nn.ModuleDict({name: nn.Embedding(n, d_model) for name, n in model_levels.items()})
        self.nuisance_shift = nn.ModuleDict({
            name: NuisanceShift(n, d_model, reference[name]) for name, n in model_levels.items() if modes[name] != "inject"
        })
        # The extra last level is the null label used for inject_dropout.
        self.nuisance_inject = nn.ModuleDict({
            name: nn.Embedding(n + 1, d_model) for name, n in model_levels.items() if modes[name] != "additive"
        })
        self.nuisance_adversary = nn.ModuleDict({
            name: NuisanceAdversary(d_model, n, self.nuisance_spec.get("conditions", {}))
            for name, n in levels.items()
        } if self.nuisance_spec.get("adversary", True) else {})
        self.pert_exp_encoder = PertExpEncoder(
            d_model, self.pert_dim, extra_dim=self.noise_dim + d_model * len(self.nuisance_inject),
        )
        self.pert_lookup = KnockoutLookup(pert_lookup, n_pert, pert_dim, pert_lookup_hide, pert_lookup_self_bonus, pert_lookup_hard) \
            if pert_lookup is not None else None
        seen = torch.zeros(n_pert, dtype=torch.bool)
        if pert_seen is not None:
            seen[pert_seen] = True
        self.register_buffer("pert_seen", seen if pert_seen is not None else None)
        self.pert_unknown_emb = nn.Parameter(torch.zeros(pert_dim)) if pert_seen is not None else None
        if self.next_group_delta:
            batch_terms = {"n_lines": levels["background"], "n_pools": levels["pool"],
                           "line_reference": reference["background"]} if decoder_terms else None
            self.next_decoder = GroupDeltaDecoder(d_model, pert_dim, wt_rule=self.next_group_delta_wt_rule,
                                                  shift_rule=next_group_delta_shift, batch_terms=batch_terms,
                                                  single=next_group_delta_single)
        elif self.next_delta:
            self.next_decoder = NextDeltaDecoder(d_model, relative=self.next_delta_control is not None)
        # the following is the perturbation decoder
        #n_pert = kwargs.get("n_perturb", 1) 
        #nlayers_pert = kwargs.get("nlayers_perturb", 3) 
        self.pert_decoder = PerturbationDecoder(d_model, n_pert, nlayers=nlayers_pert)
        # added: batch2 encoder, especially to model different cellular systems like cell line vs primary cells
        self.batch2_pad_id = None #kwargs.get("batch2_pad_id") if "batch2_pad_id" in kwargs else 2
        #self.batch2_encoder = nn.Embedding(2, d_model, padding_idx=self.batch2_pad_id)
        self.batch2_encoder = Batch2LabelEncoder(2, d_model) # should replace 2 to n_batch later
        self.n_pert = n_pert
        self.n_cls = kwargs.get("n_cls", 1) 
        
        
        if self.use_fast_transformer:
            nlayers = self.nlayers
            d_hid = self.d_hid
            nhead = self.nhead
            if self.fast_transformer_backend == 'flash':
                try:
                    from .modules import FlashTransformerEncoderLayerVarlen
                    
                    encoder_layers = FlashTransformerEncoderLayerVarlen(
                        d_model,
                        kwargs.get('nhead', nhead),
                        kwargs.get('d_hid', d_hid),
                        self.dropout,
                        batch_first=True,
                        norm_scheme=self.norm_scheme
                    )
                except Exception as e:
                    print(e)
                    print('DAO flash attention setup failed')
                    self.fast_transformer_backend = 'sdpa'

            if self.fast_transformer_backend == 'sdpa':
                print('trying pytorch SDPA')
                try:
                    from .modules import SDPATransformerEncoderLayer
                    encoder_layers = SDPATransformerEncoderLayer(
                        d_model,
                        nhead,
                        d_hid,
                        self.dropout,
                        batch_first=True,
                        norm_scheme=self.norm_scheme,
                    )
                except Exception as ee: 
                    print(ee)
                    print('pytorch sdpa attention setup failed, falling back to native pytorch attention')
                    self.use_fast_transformer = False
                    self.fast_transformer_backend = 'vanilla'
                    encoder_layers = TransformerEncoderLayer(
                        d_model, nhead, d_hid, self.dropout, batch_first=True
                    )
            self.transformer_encoder = TransformerEncoder(encoder_layers,  nlayers, enable_nested_tensor=False)

        
        # added: adding PS score decoder
        #self.n_ps = kwargs.get("n_ps") if "n_ps" in kwargs else 0
        self.n_ps = n_ps
        if self.n_ps > 0:
            self.ps_decoder = PSDecoder(d_model, self.n_ps, nlayers = nlayers_pert)
        else:
            self.ps_decoder = None
        if self.pred_lochness_next:
            self.ps_decoder2 = PSDecoder(
                d_model,
                1,
                nlayers = ps_decoder2_nlayer,
                geneinput = pert_dim,  # must match pert_encoder output width (d_model when pert_dim is None)
            )
        else:
            self.ps_decoder2 = None

    # rewrite encode function
    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
        input_pert_flags: Optional[Tensor] = None,
        nuisance_labels: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        #print('_encode batch labels:')
        #print(batch_labels)
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        # add additional perturbs
        if input_pert_flags is not None:
            perts = encode_perturbation_set(self.pert_encoder, input_pert_flags, self.pert_aggregation)  # (batch, seq_len, embsize)
            #import pdb; pdb.set_trace()
            perts_expand = perts.unsqueeze(1).repeat(1, total_embs.shape[1], 1)
            total_embs = total_embs + perts_expand

        # The source cell's nuisance labels, so the encoder can subtract them; zero at the reference
        # level, so a cell given no labels is encoded as the reference.
        if nuisance_labels is not None:
            for name, embedding in self.nuisance_input.items():
                labels = nuisance_labels[name]
                not_reference = (labels != self.nuisance_spec["reference"][name]).unsqueeze(1)
                total_embs = total_embs + (embedding(labels) * not_reference).unsqueeze(1)

        # batch2 TODO: use batch_encoder instead
        if batch_labels is not None:
            batch2_embs = self.batch2_encoder(batch_labels)
            #import pdb; pdb.set_trace()
            batch2_embs = batch2_embs.unsqueeze(1).repeat(1, total_embs.shape[1], 1)
            total_embs = total_embs + batch2_embs

        # dsbn and batch normalization
        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)


        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def _next_states(self, cell_emb: Tensor, pert_emb_next: Tensor, pert_labels_next: Optional[Tensor],
                     nuisance_labels_next: Optional[Dict[str, Tensor]], n_draws: int,
                     pert_labels_control: Optional[Tensor] = None) -> tuple:
        """Predicted target states, one per noise draw, the perturbation shift of the first draw and, with
        next_delta_control, each draw's control state (u with the control's perturbation embedding, the same
        source, injected labels and z; empty otherwise).

        The shift (None without pert_labels_control) is u minus u with the control's perturbation
        embedding, the same source and the same z: what the perturbation changes for this cell, before
        the target's nuisance shifts.

        u = pert_exp_encoder([cell_emb, pert_emb_next, injected target nuisance labels, z]) and
        h = u + the target's nuisance shifts, each read from its detached context: the source
        embedding, u itself ("state"), or the WT counterfactual w, u with the control's
        perturbation embedding and the same z ("wt_counterfactual").
        """
        n = cell_emb.shape[0]
        reference = self.nuisance_spec.get("reference", {})
        inject = []
        for name, embedding in self.nuisance_inject.items():
            labels = reference[name] * torch.ones(n, dtype=torch.long, device=cell_emb.device) \
                if nuisance_labels_next is None else nuisance_labels_next[name]
            if self.training and self.nuisance_spec.get("mode", {}).get(name) == "both":
                null = torch.rand(n, device=cell_emb.device) < self.nuisance_spec.get("inject_dropout", 0.5)
                labels = torch.where(null, embedding.num_embeddings - 1, labels)
            inject.append(embedding(labels))
        contexts = self.nuisance_spec.get("context", {})
        if "wt_counterfactual" in contexts.values():
            shape = (n,) if pert_labels_next is None or pert_labels_next.dim() == 1 else (n, 1)
            control = torch.full(shape, self.nuisance_spec["control_index"], dtype=torch.long, device=cell_emb.device)
            pert_emb_wt = encode_perturbation_set(self.pert_encoder, control, self.pert_aggregation)
        states, pert_shift, control_states = [], None, []
        if self.next_delta_control is not None:
            control = torch.full((n,), self.next_delta_control, dtype=torch.long, device=cell_emb.device)
            pert_emb_delta_control = encode_perturbation_set(self.pert_encoder, control, self.pert_aggregation)
        for _ in range(n_draws):
            noise = [torch.randn(n, self.noise_dim, device=cell_emb.device, dtype=cell_emb.dtype)] if self.noise_dim else []
            state = self.pert_exp_encoder(torch.cat([cell_emb, pert_emb_next, *inject, *noise], dim=1))
            if not states and pert_labels_control is not None:
                pert_emb_control = encode_perturbation_set(self.pert_encoder, pert_labels_control, self.pert_aggregation)
                pert_shift = state - self.pert_exp_encoder(torch.cat([cell_emb, pert_emb_control, *inject, *noise], dim=1))
            if self.next_delta_control is not None:
                control_states.append(self.pert_exp_encoder(torch.cat([cell_emb, pert_emb_delta_control, *inject, *noise], dim=1)))
            if nuisance_labels_next is not None and len(self.nuisance_shift) > 0:
                context_of = {"source": cell_emb, "state": state}
                if "wt_counterfactual" in contexts.values():
                    context_of["wt_counterfactual"] = self.pert_exp_encoder(torch.cat([cell_emb, pert_emb_wt, *inject, *noise], dim=1))
                shifted = state
                for name, shift in self.nuisance_shift.items():
                    shifted = shifted + shift(nuisance_labels_next[name], context_of[contexts.get(name, "source")].detach())
                state = shifted
            states.append(state)
        return states, pert_shift, control_states

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None, 
        pert_labels_next: Optional[Tensor] = None, 
        sf: Optional[Tensor] = None,
        sf_next: Optional[Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        PERTPRED: bool = False,
        do_sample: bool = False,
        PSPRED: bool = False,
        mvc_src: Tensor = None,
        pert_embeddings_next: Optional[Tensor] = None,
        nuisance_labels: Optional[Dict[str, Tensor]] = None,
        nuisance_labels_next: Optional[Dict[str, Tensor]] = None,
        condition_labels: Optional[Dict[str, Tensor]] = None,
        n_noise_draws: int = 1,
        pert_labels_control: Optional[Tensor] = None,
        mvc_src_values: Optional[Tensor] = None,
        celltype_labels: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            pert_labels (:obj:`Tensor`): perturbation labels, shape [batch_size]
            pert_labels_next (:obj:`Tensor`): perturbation labels for prediction, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.
            PERTPRED (:obj:`bool`): if True, return the perturbation prediction
                (PERTPRED) output. 
            PSPRED (:obj:`bool`): if True, return the PS score prediction 
                (PERTPRED) output. 
            nuisance_labels / nuisance_labels_next ({name: Tensor}): the source / target cells'
                nuisance labels (see nuisance_spec); None means the reference level.
            condition_labels ({name: Tensor}): the source cells' labels given to the adversaries.
            n_noise_draws (int): next states predicted with independent noise vectors; draws after
                the first are returned as "mvc_output_next_draws" (with MVC).
            mvc_src_values (:obj:`Tensor`): the source cells' log1p expression on the mvc_src genes, the
                base of the next-cell prediction with next_delta.

        Returns:
            dict of output Tensors.
        """
        #print('forward batch labels:')
        #print(batch_labels)
        # call the super forward function
        #output = super().forward(
        #    src,
        #    values,
        #    src_key_padding_mask,
        #    batch_labels=batch_labels,
        #    CLS=CLS,
        #    CCE=CCE,
        #    MVC=MVC,
        #    ECS=ECS,
        #    do_sample=do_sample,
        #)

        # or, rewrite the forward function
        
        transformer_output_0 = self._encode(
            src, values, src_key_padding_mask, batch_labels,
            input_pert_flags= pert_labels, # Do we use pert_flags for transformer input?
            nuisance_labels=nuisance_labels,
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        if pert_labels is not None :
            pert_emb = encode_perturbation_set(self.pert_encoder, pert_labels, self.pert_aggregation)
            # transformmer output concatenate ?
            # note only input pert_labels should be concatenated, not pert_label_next
            #import pdb; pdb.set_trace()
            #tf_o_concat=torch.cat(
            #    [
            #        transformer_output_0,
            #        pert_emb.unsqueeze(1).repeat(1, transformer_output_0.shape[1], 1),
            #   ],
            #    dim=2,
            #)
            #transformer_output=self.pert_exp_encoder(tf_o_concat)
        else:
            #tf_o_concat = None # a placeholder
            pert_emb = None
        
        transformer_output=transformer_output_0
            
        output = {}
        output["transformer_output"] = transformer_output
        output["contrastive_dict"] = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        # zero_probs is actually non-zero probability for the Bernoulli
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb_orig = self._get_cell_emb_from_layer(transformer_output, values)        
        output["contrastive_dict"]['orig_emb0'] = cell_emb_orig
        # Reconstruction of the source cell: its own nuisance shifts, read from its own embedding.
        self_state = cell_emb_orig
        if nuisance_labels is not None:
            for name, shift in self.nuisance_shift.items():
                self_state = self_state + shift(nuisance_labels[name], cell_emb_orig.detach())
        #  concatenate cell embedding with perturbation embedding to generate next cell embedding
        if pert_labels_next is not None or pert_embeddings_next is not None:
            #import pdb; pdb.set_trace()
            if pert_embeddings_next is None:
                pert_emb_next = encode_perturbation_set(self.pert_encoder, pert_labels_next, self.pert_aggregation)
                if self.pert_lookup is not None:
                    looked_up = self.pert_lookup(pert_emb_next, self.pert_encoder(self.pert_lookup.table), pert_labels_next)
                    is_control = (pert_labels_next == self.next_delta_control).unsqueeze(1)
                    pert_emb_next = torch.where(is_control, pert_emb_next, looked_up)
                if self.pert_unknown_emb is not None:
                    unseen = ~self.pert_seen[pert_labels_next] & (pert_labels_next != self.next_delta_control)
                    pert_emb_next = torch.where(unseen.unsqueeze(1), self.pert_unknown_emb.to(pert_emb_next.dtype), pert_emb_next)
            elif pert_embeddings_next.dim() == 3:
                pert_emb_next = reduce_perturbation_embeddings(pert_embeddings_next, pert_labels_next, self.pert_aggregation)
            else:
                pert_emb_next = pert_embeddings_next
            expected_pert_dim = self.d_model if self.pert_dim is None else self.pert_dim
            if pert_emb_next.shape != (src.shape[0], expected_pert_dim):
                raise ValueError("pert_embeddings_next must have shape (batch_size, pert_dim)")
            next_states, pert_shift, control_states = self._next_states(cell_emb_orig, pert_emb_next, pert_labels_next, nuisance_labels_next,
                                                        n_noise_draws, pert_labels_control)
            cell_emb_next = next_states[0]
            if pert_shift is not None:
                output["pert_shift"] = pert_shift
            output["contrastive_dict"]['next_emb0'] = cell_emb_next
        else:
            next_states, control_states = [cell_emb_orig], []
            cell_emb_next=cell_emb_orig
        
        cell_emb = cell_emb_orig
        output["cell_emb"] = cell_emb
        output["cell_emb_next"] = cell_emb_next

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
            output["cls_output_next"] = self.cls_decoder(cell_emb_next)  # (batch, n_cls)

        cur_gene_token_embs = self.encoder(mvc_src) if mvc_src is not None else self.cur_gene_token_embs
        if MVC and hasattr(self, 'mvc_decoder'):
            mvc_output = self.mvc_decoder(
                self_state
                if not self.use_batch_labels
                else torch.cat([self_state, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                cur_gene_token_embs,
                target_size_factor = sf if self.distribution is not None else None
            )
            def decode_next(state, control_state=None):
                if self.next_group_delta:
                    is_control = pert_labels_next == self.next_delta_control
                    wt_mean = self.context_wt_mean[celltype_labels] if self.next_group_delta_wt_rule else None
                    labels = {}
                    if self.next_decoder.batch_terms is not None and nuisance_labels is not None and nuisance_labels_next is not None:
                        labels = dict(line_next=nuisance_labels_next["background"], pool_source=nuisance_labels["pool"],
                                      pool_next=nuisance_labels_next["pool"])
                    return self.next_decoder(cell_emb_orig, pert_emb_next, cur_gene_token_embs, mvc_src_values, is_control, wt_mean, **labels)
                if self.next_delta:
                    if control_state is not None:
                        state = state - control_state
                    return self.next_decoder(state, cur_gene_token_embs, mvc_src_values)
                return self.mvc_decoder(
                    state if not self.use_batch_labels else torch.cat([state, batch_emb], dim=1),
                    cur_gene_token_embs,
                    target_size_factor=sf_next if self.distribution is not None else None,
                )
            mvc_output_next = decode_next(cell_emb_next, control_states[0] if control_states else None)
            if self.explicit_zero_prob and False: # bernoulli sampling is meaningless here
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]

                bernoulli_n = Bernoulli(probs=mvc_output_next["zero_probs"])
                output["mvc_output_next"] = bernoulli.sample() * mvc_output_next["pred"]
            else:
                output["mvc_output"] = mvc_output  # (batch, seq_len)
                output["mvc_output_next"] = mvc_output_next  # (batch, seq_len)
            if self.training and self.pert_unknown_emb is not None and self.next_group_delta:
                # Every source cell with the unknown knockout, for the training loss against its cell type's average change.
                unknown = self.pert_unknown_emb.to(pert_emb_next.dtype).expand(len(pert_emb_next), -1)
                no_control = torch.zeros_like(pert_labels_next, dtype=torch.bool)
                wt_mean = self.context_wt_mean[celltype_labels] if self.next_group_delta_wt_rule else None
                output["unknown_D"] = self.next_decoder(cell_emb_orig, unknown, cur_gene_token_embs, mvc_src_values, no_control, wt_mean)["D"]
            if len(next_states) > 1:
                output["mvc_output_next_draws"] = [mvc_output_next] + [
                    decode_next(state, control_states[i] if control_states else None) for i, state in enumerate(next_states[1:], 1)
                ]

        if ECS and hasattr(self, 'sim'):
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        if nuisance_labels is not None and len(self.nuisance_adversary) > 0:
            output["nuisance_adversary"] = {
                name: adversary(cell_emb_orig, condition_labels or {}) for name, adversary in self.nuisance_adversary.items()
            }


        # get cell embedding
        if PERTPRED:
            #cell_emb = output["cell_emb"]
            output["pert_output"] = self.pert_decoder(cell_emb)  # (batch, n_cls)
            output["pert_output_next"] = self.pert_decoder(cell_emb_next)  # (batch, n_cls)

        # PS score prediction
        if PSPRED and self.ps_decoder is not None:
            output["ps_output"] = self.ps_decoder(cell_emb)
            if self.pred_lochness_next:
                tf_concat=torch.cat([cell_emb_orig, pert_emb_next],dim=1)
                output["ps_output_next"] = self.ps_decoder2(tf_concat)  # this is the concatenation of cell embedding and predictive label (next)
            else:
                output["ps_output_next"] = self.ps_decoder(cell_emb_next)  # (batch, n_cls)
        return output

    @torch.no_grad()
    def encode_batch_with_perturb(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None, # the first perturbation
        pert_labels_next: Optional[Tensor] = None, # the second perturbation
        sf: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
        predict_expr = False,
        mvc_src: Tensor = None, # optional MVC tensor of gene ids for MVC decoder
        sample: bool = False, 
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        revised scgpt.TransformerModel.encode_batch but with additional perturbation
        prediction output
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the transformer token index to return. If None,
                return the full transformer output.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor tuple containing cell embeddings and model predictions.
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)
        outputs_next = array_func(shape, dtype=float32_)
        # added for perturbation predictions
        shape_perts = (N, self.n_pert)
        pert_outputs = array_func(shape_perts, dtype=float32_)
        
        # add for cls predictions
        shape_cls = (N, self.n_cls)
        cls_outputs = array_func(shape_cls, dtype=float32_)

        # added for PS score predictions
        shape_ps = (N, self.n_ps)
        ps_outputs =  array_func(shape_ps, dtype=float32_)

        if self.pred_lochness_next:
            shape_ps_next =  (N, 1) 
        else:
            shape_ps_next = shape_ps
        ps_outputs_next = array_func(shape_ps_next, dtype=float32_)
        
        expr_dict = {}
        if predict_expr:
            mlm_expr_shape = (N, src.size(1)) 
            mvc_expr_shape = (N, src.size(1)) if mvc_src is None else (N, mvc_src.size(1))
            mlm_outputs, mlm_zero_outputs = array_func(mlm_expr_shape, dtype=float32_), array_func(mlm_expr_shape, dtype=float32_)
            mvc_outputs, mvc_zero_outputs = array_func(mvc_expr_shape, dtype=float32_), array_func(mvc_expr_shape, dtype=float32_)
            mvc_next_outputs, mvc_next_zero_outputs = array_func(mvc_expr_shape, dtype=float32_), array_func(mvc_expr_shape, dtype=float32_)
            if self.distribution in ['nb', 'hnb', 'zinb', 'zig']:
                mvc_param2_outputs, mvc_next_param2_outputs = array_func(mvc_expr_shape, dtype=float32_), array_func(mvc_expr_shape, dtype=float32_)
        for i in trange(0, N, batch_size):
            src_d = src[i : i + batch_size].to(device)
            values_d = values[i : i + batch_size].to(device)
            src_key_padding_mask_d = src_key_padding_mask[i : i + batch_size].to(device)
            batch_labels_d = batch_labels[i : i + batch_size].to(device) if batch_labels is not None else None
            pert_labels_d = pert_labels[i : i + batch_size].to(device) if pert_labels is not None else None
            pert_labels_next_d = pert_labels_next[i : i + batch_size].to(device) if pert_labels_next is not None else None
            mvc_src_d = mvc_src[i : i + batch_size].to(device) if mvc_src is not None else None
            sf_d = sf[i : i + batch_size].to(device) if sf is not None else None
            batch_output = self(
                src_d,
                values_d,
                src_key_padding_mask_d,
                batch_labels=batch_labels_d,
                pert_labels=pert_labels_d,
                pert_labels_next=pert_labels_next_d,
                sf=sf_d,
                sf_next=sf_d,
                CLS=True,
                MVC=predict_expr,
                PERTPRED=True,
                PSPRED=self.ps_decoder is not None,
                mvc_src=mvc_src_d,
            )
            raw_output = batch_output["transformer_output"].detach()
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

            cell_emb_next = batch_output["cell_emb_next"].detach()
            if output_to_cpu:
                cell_emb_next = cell_emb_next.cpu()
            if return_np:
                cell_emb_next = cell_emb_next.numpy()
            outputs_next[i : i + batch_size] = cell_emb_next if pert_labels_next_d is not None else output
            
            pert_output = batch_output["pert_output"].detach()
            if output_to_cpu:
                pert_output = pert_output.cpu()
            if return_np:
                pert_output = pert_output.numpy()
            pert_outputs[i : i + batch_size] = pert_output

            cls_output = batch_output["cls_output"].detach()
            if output_to_cpu:
                cls_output = cls_output.cpu()
            if return_np:
                cls_output = cls_output.numpy()
            cls_outputs[i : i + batch_size] = cls_output

            # always check if ps decoder is used or not
            if self.ps_decoder is not None:
                ps_output = batch_output["ps_output"].detach()
                if output_to_cpu:
                    ps_output = ps_output.cpu()
                if return_np:
                    ps_output = ps_output.numpy()
                ps_outputs[i : i + batch_size] = ps_output
                ps_output_next = (
                    batch_output["ps_output_next"].detach()
                    if self.pred_lochness_next
                    else batch_output["ps_output"].detach()
                )
                if output_to_cpu:
                    ps_output_next = ps_output_next.cpu()
                if return_np:
                    ps_output_next = ps_output_next.numpy()
                ps_outputs_next[i : i + batch_size] = ps_output_next
            
            if predict_expr:
                mlm_pred = batch_output['mlm_output'].detach()
                mlm_zero_probs = batch_output.get('mlm_zero_probs', torch.ones_like(mlm_pred)).detach()
                if output_to_cpu:
                    mlm_pred, mlm_zero_probs = mlm_pred.cpu(), mlm_zero_probs.cpu()
                if return_np:
                    mlm_pred, mlm_zero_probs = mlm_pred.numpy(), mlm_zero_probs.numpy()
                mvc_generator = DistributionGenerator(self.distribution)
                mvc_output = mvc_generator.generate(batch_output["mvc_output"], sample = sample, to_numpy = return_np, device = next(self.parameters()).device)
                mvc_output_next = mvc_generator.generate(batch_output["mvc_output_next"], sample = sample, to_numpy = return_np, device = next(self.parameters()).device)
                if output_to_cpu and not return_np:
                    mvc_output = {key: value.cpu() if torch.is_tensor(value) else value for key, value in mvc_output.items()}
                    mvc_output_next = {key: value.cpu() if torch.is_tensor(value) else value for key, value in mvc_output_next.items()}
                mvc_pred, mvc_param2, mvc_zero_probs = mvc_output['pred'], mvc_output['param2'], mvc_output['zero_probs']
                mvc_pred_next, mvc_param2_next, mvc_zero_probs_next = mvc_output_next['pred'], mvc_output_next['param2'], mvc_output_next['zero_probs']
                mlm_outputs[i : i + batch_size], mvc_outputs[i : i + batch_size], mvc_next_outputs[i : i + batch_size] = mlm_pred, mvc_pred, mvc_pred_next
                if mvc_param2 is not None and mvc_param2_next is not None:
                    mvc_param2_outputs[i : i + batch_size], mvc_next_param2_outputs[i : i + batch_size]  = mvc_param2, mvc_param2_next
                if self.explicit_zero_prob:
                    mlm_zero_outputs[i : i + batch_size], mvc_zero_outputs[i : i + batch_size], mvc_next_zero_outputs[i : i + batch_size] = mlm_zero_probs, mvc_zero_probs,  mvc_zero_probs_next
                

        if predict_expr:
            expr_dict['mlm_expr'], expr_dict['mvc_expr'], expr_dict['mvc_next_expr'] = (mlm_outputs[:,1:], mvc_outputs[:,1:], mvc_next_outputs[:,1:])
            if self.explicit_zero_prob:
                expr_dict['mlm_expr_zero'], expr_dict['mvc_expr_zero'], expr_dict['mvc_next_expr_zero'] = (mlm_zero_outputs[:,1:], mvc_zero_outputs[:,1:], mvc_next_zero_outputs[:,1:])
            if self.distribution in ['nb', 'hnb', 'zinb', 'zig']:
                expr_dict['mvc_param2'], expr_dict['mvc_next_param2'] = (mvc_param2_outputs[:, 1:], mvc_next_param2_outputs[:, 1:])
        return outputs, outputs_next, pert_outputs, cls_outputs, ps_outputs, ps_outputs_next, expr_dict
