from typing import Dict, Mapping, Optional, Tuple, Any, Union
import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention import SDPBackend
# Try to import the MHA module from flash-attn v2
FLASH_ATTENTION_VERSION = None
flash_attn_qkvpacked_func = None
flash_attn_varlen_func = None

# 2. Try to import the newest version first
try:
    # Assuming 'flash_attn_interface' is the newer package/module
    from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_func
    FLASH_ATTENTION_VERSION = '3'
    print("✅ Detected Flash Attention v3.")
except ImportError:
    # 3. If the first import fails, try the next one
    try:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func
        FLASH_ATTENTION_VERSION = '2'
        print("✅ Detected Flash Attention v2.")
    except ImportError:
        # 4. If all imports fail, provide a notice
        print("⚠️ Flash Attention not installed. Model will use standard attention.")


class Empty:
    pass

class FlashTransformerEncoderLayerVarlen(nn.Module):
    """
    Alternative implementation that uses flash_attn_varlen_func for better handling
    of sequences with different lengths (padding).
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
        causal=False,
        bias = True,
        use_flash_attn = True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if FLASH_ATTENTION_VERSION is None:
            raise ImportError(
                "FlashTransformerEncoderLayerVarlen requires flash-attn (v2: 'flash_attn', v3: 'flash_attn_interface'), "
                "but neither could be imported. Install flash-attn or use fast_transformer_backend='sdpa'."
            )
        self.flash_version = FLASH_ATTENTION_VERSION
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal
        
        # Multi-head attention components
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.self_attn = Empty() # Dummy code because TransformerEncoder expects self.self_attn.batch_first
        self.self_attn.batch_first = batch_first
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def _compute_packing_info(self, key_padding_mask):
        """
        Pre-compute all packing information to avoid expensive loops.
        
        Args:
            key_padding_mask: Boolean mask of shape (batch_size, seq_len)
                             True indicates positions to be masked
        
        Returns:
            batch_indices: Tensor of batch indices for valid positions
            seq_indices: Tensor of sequence indices for valid positions  
            seqlens: Tensor of actual sequence lengths per batch
            cu_seqlens: Cumulative sequence lengths for flash attention
            total_valid_tokens: Total number of valid (non-padded) tokens
        """
        valid_mask = ~key_padding_mask  # True for valid positions
        # Find all valid positions at once using vectorized operations
        batch_indices, seq_indices = torch.where(valid_mask)
        
        # Compute actual sequence lengths per batch
        seqlens = valid_mask.sum(dim=1, dtype=torch.int32)
        
        # Create cumulative sequence lengths for flash attention
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=key_padding_mask.device),
            seqlens.cumsum(dim=0, dtype=torch.int32)
        ])
        
        total_valid_tokens = batch_indices.shape[0]
        
        return batch_indices, seq_indices, seqlens, cu_seqlens, total_valid_tokens

    def _pack_sequences_fast(self, tensor, batch_indices, seq_indices):
        """
        Fast packing using advanced indexing instead of loops.
        
        Args:
            tensor: Input tensor of shape (batch_size, seq_len, nhead, head_dim)
            batch_indices: Batch indices for valid positions
            seq_indices: Sequence indices for valid positions
            
        Returns:
            packed_tensor: Tensor of shape (total_valid_tokens, nhead, head_dim)
        """
        # Use advanced indexing - much faster than loops and concatenation
        return tensor[batch_indices, seq_indices]

    def _unpack_sequences_fast(self, packed_tensor, batch_indices, seq_indices, orig_shape):
        """
        Fast unpacking using direct assignment instead of loops.
        
        Args:
            packed_tensor: Packed tensor of shape (total_valid_tokens, nhead, head_dim)
            batch_indices: Batch indices for valid positions
            seq_indices: Sequence indices for valid positions
            orig_shape: Original shape (batch_size, seq_len, nhead, head_dim)
            
        Returns:
            unpacked_tensor: Tensor of original shape with results scattered back
        """
        batch_size, seq_len, nhead, head_dim = orig_shape
        
        # Initialize output tensor with zeros
        output = torch.zeros(
            orig_shape,
            dtype=packed_tensor.dtype,
            device=packed_tensor.device
        )
        
        # Use advanced indexing for fast scattering
        output[batch_indices, seq_indices] = packed_tensor
        
        return output


    def _flash_attention(self, x, key_padding_mask=None):
        """
        Perform flash attention on the input tensor using variable length attention
        when padding mask is present.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            key_padding_mask: Boolean mask of shape (batch_size, seq_len)
                             True indicates positions to be masked
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Reshape to separate Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        #print(qkv.shape)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, batch_size, seq_len, nhead, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, seq_len, nhead, head_dim)
        #print(qkv.shape)
        # Check if we have any padding
        if key_padding_mask is not None:
            # Convert to boolean if needed
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            
            # Check if there's actual padding
            if not key_padding_mask.any():
                key_padding_mask = None
        
        if key_padding_mask is None:
            # No padding mask - use the efficient packed version
            # Repack for flash_attn_qkvpacked_func
            qkv_for_flash = torch.stack([q, k, v], dim=4)  # (batch, seq_len, nhead, head_dim, 3)
            #print(qkv_for_flash.shape)
            qkv_for_flash = qkv_for_flash.permute(0, 1, 4, 2, 3)  # (batch, seq_len, 3, nheads, head_dim)
            #print(qkv_for_flash.shape)
            if self.flash_version == '3':
                attn_output = flash_attn_qkvpacked_func(
                    qkv_for_flash,
                    softmax_scale=None,
                    causal=self.causal,
                )
            elif self.flash_version == '2':
                attn_output = flash_attn_qkvpacked_func(
                    qkv_for_flash,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    softmax_scale=None,
                    causal=self.causal,
                    return_attn_probs=False,
                )
            #print(attn_output.shape)
            # attn_output is (batch, seq_len, nhead, head_dim)
        else:
            # Use variable length attention for sequences with padding
            # Calculate actual sequence lengths
            #seqlens = (~key_padding_mask).sum(dim=1, dtype=torch.int32)

            batch_indices, seq_indices, seqlens, cu_seqlens, total_valid_tokens = \
                self._compute_packing_info(key_padding_mask)

            # Handle edge case where all sequences might be fully padded
            
            # Create cumulative sequence lengths
            """
            cu_seqlens = torch.cat([
                torch.tensor([0], dtype=torch.int32, device=x.device),
                seqlens.cumsum(dim=0, dtype=torch.int32)
            ])
            
            # Create mask for valid positions
            valid_mask = ~key_padding_mask  # True for valid positions
            
            # Pack sequences by removing padded positions
            q_packed_list = []
            k_packed_list = []
            v_packed_list = []
            
            for b in range(batch_size):
                valid_indices = valid_mask[b]
                if valid_indices.any():  # Only process if there are valid tokens
                    q_packed_list.append(q[b][valid_indices])
                    k_packed_list.append(k[b][valid_indices])
                    v_packed_list.append(v[b][valid_indices])
            """
            # Handle edge case where all sequences might be fully padded
            #if q_packed_list:
            if total_valid_tokens == 0:
                # All sequences are fully padded - return zeros
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=x.dtype,
                    device=x.device
                )
            else:
                # Fast packing using vectorized operations
                q_packed = self._pack_sequences_fast(q, batch_indices, seq_indices)
                k_packed = self._pack_sequences_fast(k, batch_indices, seq_indices)
                v_packed = self._pack_sequences_fast(v, batch_indices, seq_indices)
                #q_packed = torch.cat(q_packed_list, dim=0)  # (total_valid_tokens, nhead, head_dim)
                #k_packed = torch.cat(k_packed_list, dim=0)
                #v_packed = torch.cat(v_packed_list, dim=0)
                
                # Apply variable length flash attention
                max_seqlen = int(seqlens.max().item())
                if self.flash_version == '3':
                    attn_output_packed = flash_attn_varlen_func(
                        q_packed,
                        k_packed, 
                        v_packed,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        softmax_scale=None,
                        causal=self.causal,
                    
                    )
                elif self.flash_version == '2':
                    attn_output_packed = flash_attn_varlen_func(
                        q_packed,
                        k_packed, 
                        v_packed,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        dropout_p=self.dropout1.p if self.training else 0.0,
                        softmax_scale=None,
                        causal=self.causal,
                        return_attn_probs=False,
                    )
                
                # attn_output_packed shape: (total_valid_tokens, nhead, head_dim)
                orig_shape = (batch_size, seq_len, self.nhead, self.head_dim)
                attn_output = self._unpack_sequences_fast(
                    attn_output_packed, 
                    batch_indices, 
                    seq_indices, 
                    orig_shape
                )
                
                # OLD: Unpack the output back to original shape with padding
                """
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=attn_output_packed.dtype,
                    device=attn_output_packed.device
                )
                
                # Scatter the results back to their original positions
                start_idx = 0
                for b in range(batch_size):
                    valid_indices = valid_mask[b]
                    if valid_indices.any():
                        num_valid = valid_indices.sum().item()
                        attn_output[b][valid_indices] = attn_output_packed[start_idx:start_idx + num_valid]
                        start_idx += num_valid
                
            else:
                # All sequences are fully padded
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=x.dtype,
                    device=x.device
                )
                """
                
        # Reshape output: (batch, seq_len, nhead, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.out_proj(attn_output)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                Shape: (batch_size, seq_len, d_model) if batch_first=True
            src_mask: the mask for the src sequence (optional).
                Note: FlashAttention v2 has limited support for arbitrary attention masks
            src_key_padding_mask: the mask for the src keys per batch (optional).
                Shape: (batch_size, seq_len), True means ignore/mask that position
                Can be bool or float tensor (will be converted to bool)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        
        if src_mask is not None:
            # FlashAttention v2 supports causal masks natively but arbitrary masks need special handling
            if not self.causal:
                raise ValueError(
                    "FlashAttention v2 only supports causal masks natively. "
                    "For arbitrary attention masks, consider using standard attention."
                )
        
        # Ensure batch_first format
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        if self.norm_scheme == "pre":
            # Pre-normalization
            src = self.norm1(src)
            src2 = self._flash_attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            # Post-normalization
            src2 = self._flash_attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Convert back if needed
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src
    

# Try to use other backends for attention
class SDPATransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder Layer that uses torch.nn.functional.scaled_dot_product_attention
    to automatically leverage the best available attention backend (e.g., FlashAttention, cuDNN).
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",
        causal=False,
        bias = True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal
        self.self_attn = Empty()
        self.self_attn.batch_first = batch_first
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # REMOVED: The FlashSelfAttention module and use_flash_attn flag are no longer needed.
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Layer normalization and dropouts
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    # SIMPLIFIED: Replaced the complex _flash_attention method with a cleaner one.
    def _attention(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Perform attention using PyTorch's scaled_dot_product_attention.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Reshape and permute for SDPA
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, nhead, seq_len, head_dim)
        
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask follows the torch TransformerEncoder convention:
            # True = padded token, ignore it.
            # SDPA's boolean attn_mask uses the opposite convention:
            # True = this key may be attended to.
            # Invert, then broadcast to (batch, nhead, q_len, k_len).
            attn_mask = (~key_padding_mask.bool()).view(batch_size, 1, 1, seq_len)


        # Let PyTorch pick the best available kernel (flash, memory-efficient,
        # cuDNN, or math). Restricting to a single backend raises at forward
        # time when that kernel is unsupported for the current torch build,
        # GPU, dtype, or mask, and that failure cannot be caught by the
        # construction-time try/except in pertTF.py.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout1.p if self.training else 0.0,
            is_causal=self.causal,
        )
            
        # Reshape output back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if src_mask is not None and not self.causal:
            raise ValueError("SDPATransformerEncoderLayer only supports a causal mask via the 'causal' flag.")
        
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        if self.norm_scheme == "pre":
            src_norm = self.norm1(src)
            attn_out = self._attention(src_norm, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            
            src_norm = self.norm2(src)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
            src = src + self.dropout2(ff_out)
        else: # post-norm
            attn_out = self._attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            src = self.norm1(src)
            
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(ff_out)
            src = self.norm2(src)
        
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src


"""
All Modules other than the flash attention modules are not yet imported in pertTF.py
This script hopes to refactor the module classes and provide better control and customization 
"""

class PerturbationDecoder(nn.Module):
    """
    Decoder for perturbation label prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_pert)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class PSDecoder(nn.Module):
    """
    Decoder for ps score prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
        geneinput: int = 0,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()

        self.input_dim =  d_model + geneinput # * 2 #this is a concatenation of cell embedding and perturbation embedding

        
        for i in range(nlayers - 1):
            if i == 0:
                self._decoder.append(nn.Linear(self.input_dim, d_model))
            else:
                self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_pert)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class Batch2LabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x

class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class PertLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        # One row per perturbation, so a row is only updated in the batches that contain it.
        # perttf.utils.set_optimizer gives such tables their own, smaller Adam eps.
        self.embedding.sparse_parameter_updates = True
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class FeaturePertEncoder(nn.Module):
    """
    Feature-conditioned perturbation encoder.

    Drop-in replacement for ``PertLabelEncoder`` with an identical
    ``forward(LongTensor idx) -> (batch, embedding_dim)`` interface, so the rest of
    ``PerturbationTFModel`` needs no change. Instead of looking up an independently
    learned row per genotype index (``nn.Embedding``), it maps a *fixed feature
    vector* for each perturbation through a small trainable MLP. Because the
    embedding is a smooth function of features (gene/drug identity), a perturbation
    that was NEVER seen in training still receives a meaningful embedding from its
    feature row -> this is what enables zero-shot (regime-A) prediction of novel
    conditions, which the index-lookup encoder structurally cannot do.

    Args:
        pert_features: array-like, shape (n_pert, feat_dim). Row i is the feature
            vector for the perturbation whose ``genotype_to_index`` value is i.
            Held-out/novel conditions MUST be assigned an index + row here too.
        embedding_dim: output embedding size (must match ``pert_dim`` / d_model use).
        hidden_dim: MLP hidden width (defaults to embedding_dim).
        padding_idx: optional index whose embedding is forced to zero (e.g. a pad
            perturbation), mirroring nn.Embedding's padding_idx semantics.
    """

    def __init__(
        self,
        pert_features,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        feats = torch.as_tensor(pert_features, dtype=torch.float32)
        if feats.dim() != 2:
            raise ValueError(f"pert_features must be 2-D (n_pert, feat_dim), got {tuple(feats.shape)}")
        # fixed (non-trained) feature table; moves with .to(device) and is saved in state_dict
        self.register_buffer("features", feats)
        feat_dim = feats.shape[1]
        hidden_dim = embedding_dim if hidden_dim is None else hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.padding_idx = padding_idx

    def forward(self, x: Tensor) -> Tensor:
        feats = self.features[x]          # (batch, feat_dim) — gathers rows by index
        emb = self.proj(feats)            # (batch, embedding_dim)
        emb = self.enc_norm(emb)
        if self.padding_idx is not None:
            emb = emb.masked_fill((x == self.padding_idx).unsqueeze(-1), 0.0)
        return emb


class PertExpEncoder(nn.Module):
    """
    Concatenating gene expression embeddings (from transformers) with perturbation embeddings (from scGPT's PertEncoder)
    """
    def __init__(
        self,
        d_model: int,
        pert_dim: int = None,
        extra_dim: int = 0,
    ):
        super().__init__()
        pert_dim = d_model if pert_dim is None else pert_dim
        # extra_dim: further inputs concatenated after the perturbation embedding (noise, injected nuisance labels)
        d_in = d_model + pert_dim + extra_dim
        #d_in = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.Sigmoid(),#nn.ReLU(),#nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(d_model, d_model),
            #nn.LayerNorm(d_model),
            #nn.Linear(d_model, d_model),
        )


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer concatenated with perturbation embedding, (batch, d_model*2)"""
        # pred_value = self.fc(x).squeeze(-1)  
        return self.fc(x) # (batch, d_model)



class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
# added here for potential customisations
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

# added here for potential customisations
class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

# added here for potential customisations
class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = True, 
        distribution: str = None,  # Options: 'nb', 'zinb', 'hnb', 'zig', 'pois', 'zipois'
        use_batch_labels: bool = False,
        sf_scaling: bool = True,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
            explicit_zero_prob (:obj:`bool`): whther to modle the NON-ZERO probability
            distribution (:obj:`str`): expression distribution 
                'nb' (negative binomial) 
                'zinb' (zero-inflated negative binomial) 
                'hnb' (hurdle truncated negative binomial) 
                'zig' (zero inflated gaussian) 
                'pois' (poisson)
                'zipois' (zero inflated poisson)
            use_batch_labels (:obj:`bool`): whether batch label is included during modeling
        """
        super().__init__()
        self.distribution = None if distribution is None else distribution.lower()
        self.sf_scaling = sf_scaling
        valid_dists = ['nb', 'zinb', 'hnb', 'zig', 'pois', 'zipois', None]
        if self.distribution not in valid_dists:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # 1. Determine Gate (Pi)
        self.explicit_zero_prob = self.distribution in ['zinb', 'hnb', 'zig', 'zipois'] or explicit_zero_prob
        
        # 2. Determine param2 (Theta/Sigma)
        self.has_param2 = self.distribution in ['nb', 'zinb', 'hnb', 'zig']

        self.expr_act = nn.Identity() if distribution is None else nn.Softplus()
        d_in = d_model * 2 if use_batch_labels else d_model
        
        if self.has_param2:
            self.gene2param2 = torch.nn.Linear(d_model, 1) 

        # --- Architecture Setup (Same as before) ---
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if self.explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
            if self.explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
            if self.explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")
        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")

    def forward(
        self, 
        cell_emb: Tensor, 
        gene_embs: Tensor,
        target_size_factor: Tensor = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            cell_emb: (batch, d_model)
            gene_embs: (batch, seq_len, d_model)
            target_size_factor: (batch, 1) - The total counts of the target cell
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        
        # 1. Calculate param2
        param2 = None
        if self.has_param2:
            param2 = F.softplus(self.gene2param2(gene_embs)).squeeze(2)
            param2 = 1/(param2 + 1e-6)
            param2 = torch.clamp(param2, min=1e-4, max=1e4)

        # 2. Calculate Prediction
        pred_concentration = None
        zero_probs = None 
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            pred_concentration = self.expr_act(pred_value)
            if self.explicit_zero_prob:
                zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
                zero_probs = torch.sigmoid(zero_logits)

        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            if self.explicit_zero_prob:
                zero_logits = torch.bmm(self.W_zero_logit(query_vecs), 
                                      self.query_activation(cell_emb.unsqueeze(2))).squeeze(2)
                zero_probs = torch.sigmoid(zero_logits)
            cell_emb_expanded = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)
            combined = torch.cat([cell_emb_expanded, query_vecs], dim=2)
            h = self.hidden_activation(self.fc1(combined))
            raw_pred = self.fc2(h).squeeze(2)
            pred_concentration = self.expr_act(raw_pred)

        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            if self.explicit_zero_prob:
                zero_logits = torch.bmm(self.W_zero_logit(query_vecs), 
                                      self.query_activation(cell_emb.unsqueeze(2))).squeeze(2)
                zero_probs = torch.sigmoid(zero_logits)
            cell_emb = cell_emb.unsqueeze(1)
            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            raw_pred = self.fc2(h).squeeze(2)
            pred_concentration = self.expr_act(raw_pred)

        # 3. Finalize Mu
        if self.distribution == 'zig' or self.distribution is None:
            mu = pred_concentration
        else:
            # maybe use constant sizefactor (sf_scaling = False) and let the model learn what it can
            target_size_factor = 1 if target_size_factor is None or not self.sf_scaling else target_size_factor
            # OR this should be division because inputs are size factor agnostic thus outputs are size factor agnostic??
            # Furthermore, testing shows lower NLL when division is used as opposed to multiplication
            mu = pred_concentration / target_size_factor 

        return {
            "pred": mu,
            "param2": param2, 
            "zero_probs": zero_probs if self.explicit_zero_prob else torch.ones_like(mu),
            "distribution": self.distribution
        }


class NextDeltaDecoder(nn.Module):
    """Next-cell expression as a change applied to the source cell's observed expression.

    On the normalised-count scale of the log1p inputs (expm1 of the source values), per gene j:
        m_next[j] = exp(delta[j]) * x_src[j] + exp(add[j])
    exp(delta) scales the expression the source has (decrease or increase); exp(add) is expression added
    where the source has little or none (a gene switched on, or one the source cell did not sample).
    delta and add are read from the next state with a query from the gene embedding, as MVCDecoder's inner
    product; add also has a gene-specific offset. At initialisation delta = 0 and exp(add) = exp(add_init),
    so the prediction starts as the source cell itself, and the mean over a group of source cells starts
    at the group's real mean. Returns pred = log1p(m_next), the scale of the log-normalised targets
    (distribution None).

    With relative=True the input is a state difference (the next state minus the control's state) and the
    added expression is exp(add) - exp(offset), so a zero difference returns the source exactly; a gene
    can then lose added expression too, and m_next is clipped at 0.
    """

    def __init__(self, d_model: int, add_init: float = -3.0, relative: bool = False):
        super().__init__()
        self.relative = relative
        self.gene2query = nn.Linear(d_model, d_model)
        self.query_activation = nn.Sigmoid()
        self.W_delta = nn.Linear(d_model, d_model, bias=False)
        self.W_add = nn.Linear(d_model, d_model, bias=False)
        self.add_offset = nn.Linear(d_model, 1)
        for layer in (self.W_delta, self.W_add, self.add_offset):
            nn.init.zeros_(layer.weight)
        nn.init.constant_(self.add_offset.bias, add_init)

    def forward(self, state: Tensor, gene_embs: Tensor, src_values: Tensor) -> Dict[str, Tensor]:
        query = self.query_activation(self.gene2query(gene_embs))   # (batch, genes, d_model)
        state = state.unsqueeze(2)
        delta = torch.bmm(self.W_delta(query), state).squeeze(2).float()
        offset = self.add_offset(query).squeeze(2).float()
        add = torch.bmm(self.W_add(query), state).squeeze(2).float() + offset
        # The CLS slot and padding hold negative values: no expression.
        source = torch.expm1(src_values.float()).clamp(min=0)
        if self.relative:
            pred = torch.log1p((torch.exp(delta) * source + torch.exp(add) - torch.exp(offset)).clamp(min=0))
        else:
            pred = torch.log1p(torch.exp(delta) * source + torch.exp(add))
        return {"pred": pred, "param2": None, "zero_probs": torch.ones_like(pred), "distribution": None}



class GroupDeltaDecoder(nn.Module):
    """Next-cell expression from one change per gene, decomposed into an overall, a context, a perturbation and an
    interaction term, applied to the source cell by a fixed rule.

    With q_g the query read from gene g's embedding (as in NextDeltaDecoder), h the source cell's embedding and p the
    target's perturbation embedding:
        D_g = mu(q_g) + q_g . W_alpha h + q_g . W_beta p + q_g . W_gamma f([h, p, h * p])
    D is a change of log1p of the group mean of normalised counts. On the source's normalised counts x (expm1 of the
    source values) the rule
        D > 0:   m = (x + 1) e^D - 1   (log1p(m) = log1p(x) + D; a cell with 0 counts gets e^D - 1)
        D <= 0:  m = x e^D             (zeros stay 0)
    keeps the mean over cells exact on the count scale when D is the same for all of them. With wt_rule, a decrease is
    instead the scaling that lowers the group's WT mean mu (forward's wt_mean) by D on the log1p scale,
        D <= 0:  m = r x,  r = ((mu + 1) e^D - 1) / mu, at least 0  (log1p(r mu) - log1p(mu) = D; r = 0 below -log1p(mu))
    so that D is the change of log1p of the group mean in both directions, whatever the gene's level (with m = x e^D, D
    is a fold change for decreases, poorly determined for low genes; section 51 of the study notes). A control target gets D = 0,
    the cell itself. With shift_rule, D is instead a change of the group's mean log1p (the scale of the cell-level metrics)
    and is added to every cell's log1p value, clipped at 0: pred = max(log1p(x) + D, 0) (the mean log1p moves by exactly D
    except where clipping cuts a decrease short; section 64 of the study notes).
    With batch_terms ({"n_lines", "n_pools", "line_reference"}), two terms for pairs that cross a line or pool are added to
    D (section 66): Lambda = q . W_line g([e_line, h]), the target line's part shared by every knockout (0 for the
    reference line and for control targets), and S = q . W_pool k([e_pool, h]), the shift from the source's pool to the
    target's (0 when the pools match). Without forward's line and pool labels both are 0.
    With single, the head is not decomposed: D = q . W_gamma f([h, p, h * p]) alone (returned as gamma; mu, alpha and beta
    are 0), the ablation of section 69 of the study notes.
    Every term starts at 0, so the prediction starts as the source cell. Returns pred = log1p(m) (the
    scale of the log-normalised targets, distribution None), D and its four terms (gamma also for its penalty).
    """

    def __init__(self, d_model: int, pert_dim: int, wt_rule: bool = False, shift_rule: bool = False,
                 batch_terms: Optional[dict] = None, single: bool = False):
        super().__init__()
        self.wt_rule = wt_rule
        self.shift_rule = shift_rule
        self.single = single
        self.gene2query = nn.Linear(d_model, d_model)
        self.query_activation = nn.Sigmoid()
        if not single:
            self.mu = nn.Linear(d_model, 1)
            self.W_alpha = nn.Linear(d_model, d_model, bias=False)
            self.W_beta = nn.Linear(d_model, pert_dim, bias=False)
        self.interaction = nn.Sequential(nn.Linear(2 * d_model + pert_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.W_gamma = nn.Linear(d_model, d_model, bias=False)
        self.batch_terms = batch_terms
        if batch_terms is not None:
            self.line_emb = nn.Embedding(batch_terms["n_lines"], d_model)
            self.pool_emb = nn.Embedding(batch_terms["n_pools"], d_model)
            self.line_net = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.pool_net = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.W_line = nn.Linear(d_model, d_model, bias=False)
            self.W_pool = nn.Linear(d_model, d_model, bias=False)
        main = (self.W_gamma,) if single else (self.mu, self.W_alpha, self.W_beta, self.W_gamma)
        for layer in main + ((self.W_line, self.W_pool) if batch_terms else ()):
            nn.init.zeros_(layer.weight)
        if not single:
            nn.init.zeros_(self.mu.bias)

    def forward(self, cell_emb: Tensor, pert_emb: Tensor, gene_embs: Tensor, src_values: Tensor,
                is_control: Tensor, wt_mean: Optional[Tensor] = None, line_next: Optional[Tensor] = None,
                pool_source: Optional[Tensor] = None, pool_next: Optional[Tensor] = None) -> Dict[str, Tensor]:
        query = self.query_activation(self.gene2query(gene_embs))   # (batch, genes, d_model)
        both = self.interaction(torch.cat([cell_emb, pert_emb, cell_emb * pert_emb], dim=1))
        # q . W v = q . (v W): each term's vector is mapped first, so no (batch, genes, d_model) product per term.
        read = lambda vector: torch.bmm(query, vector.unsqueeze(2)).squeeze(2).float()
        gamma = read(both @ self.W_gamma.weight)
        if self.single:
            mu = alpha = beta = torch.zeros_like(gamma)
        else:
            mu = self.mu(query).squeeze(2).float()
            alpha, beta = read(cell_emb @ self.W_alpha.weight), read(pert_emb @ self.W_beta.weight)
        D = mu + alpha + beta + gamma
        D = D.masked_fill(is_control.unsqueeze(1), 0.0)
        line = pool = torch.zeros_like(D)
        if self.batch_terms is not None and line_next is not None:
            on_line = (line_next != self.batch_terms["line_reference"]) & ~is_control
            line = read(self.line_net(torch.cat([self.line_emb(line_next), cell_emb], dim=1)) @ self.W_line.weight) * on_line.unsqueeze(1)
            on_pool = pool_source != pool_next
            pool = read(self.pool_net(torch.cat([self.pool_emb(pool_source), cell_emb], dim=1)) @ self.W_pool.weight) * on_pool.unsqueeze(1)
            D = D + line + pool
        # The CLS slot and padding hold negative values: no expression.
        source = torch.expm1(src_values.float()).clamp(min=0)
        # Increases on the log scale directly; the decrease branch is clamped so that neither branch overflows
        # (torch.where passes gradients through both).
        if self.shift_rule:
            pred = (torch.log1p(source) + D).clamp(min=0)
        elif self.wt_rule:
            wt = wt_mean.float()   # the group's WT mean (mu in the formula above; not the term mu)
            ratio = ((wt + 1) * torch.exp(D.clamp(max=0)) - 1) / wt.clamp(min=1e-8)
            ratio = torch.where(wt > 0, ratio, torch.ones_like(ratio)).clamp(min=0)   # no source has the gene
            pred = torch.where(D > 0, torch.log1p(source) + D, torch.log1p(source * ratio))
        else:
            pred = torch.where(D > 0, torch.log1p(source) + D, torch.log1p(source * torch.exp(D.clamp(max=0))))
        return {"pred": pred, "D": D, "mu": mu, "alpha": alpha, "beta": beta, "gamma": gamma, "line": line, "pool": pool, "param2": None, "zero_probs": torch.ones_like(pred), "distribution": None}


from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)

class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super(_DomainSpecificBatchNorm, self).__init__()
        self._cur_domain = None
        self.num_domains = num_domains
        self.bns = nn.ModuleList(
            [
                self.bn_handle(num_features, eps, momentum, affine, track_running_stats)
                for _ in range(num_domains)
            ]
        )

    @property
    def bn_handle(self) -> nn.Module:
        raise NotImplementedError

    @property
    def cur_domain(self) -> Optional[int]:
        return self._cur_domain

    @cur_domain.setter
    def cur_domain(self, domain_label: int):
        self._cur_domain = domain_label

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, domain_label: int) -> torch.Tensor:
        self._check_input_dim(x)
        if domain_label >= self.num_domains:
            raise ValueError(
                f"Domain label {domain_label} exceeds the number of domains {self.num_domains}"
            )
        bn = self.bns[domain_label]
        self.cur_domain = domain_label
        return bn(x)


class DomainSpecificBatchNorm1d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm1d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() > 3:
            raise ValueError(
                "expected at most 3D input (got {}D input)".format(input.dim())
            )


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm2d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

class AdversarialDiscriminator(nn.Module):
    """
    From scGPT
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class NuisanceShift(nn.Module):
    """A shift of the cell state caused by a nuisance label (e.g. parent background, pool group).

    shift = MLP([embedding(label), context]), where context is a state the caller detaches (the
    source embedding, the predicted state, or a WT counterfactual), so the shift can depend on the
    cell's state but nothing upstream is trained to feed it. The reference level has no shift, so
    a cell given no labels is predicted at the reference.
    """

    def __init__(self, n_levels: int, d_model: int, reference: int):
        super().__init__()
        self.reference = reference
        self.embedding = nn.Embedding(n_levels, d_model)
        self.fc = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, labels: Tensor, context: Tensor) -> Tensor:
        shift = self.fc(torch.cat([self.embedding(labels), context], dim=1))
        return shift * (labels != self.reference).unsqueeze(1).to(shift.dtype)


class NuisanceAdversary(nn.Module):
    """Predicts a nuisance label from the cell embedding through a gradient reversal, so the encoder
    is trained to hide it. Condition labels (e.g. cell type, day) are given to the classifier, so
    only nuisance information beyond what cell type and day explain is removed."""

    def __init__(self, d_model: int, n_levels: int, condition_levels: Dict[str, int]):
        super().__init__()
        self.conditions = nn.ModuleDict({name: nn.Embedding(n, d_model) for name, n in condition_levels.items()})
        self.fc = nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(), nn.Linear(d_model, n_levels))

    def forward(self, cell_emb: Tensor, condition_labels: Dict[str, Tensor]) -> Tensor:
        x = grad_reverse(cell_emb)
        for name, embedding in self.conditions.items():
            x = x + embedding(condition_labels[name])
        return self.fc(x)
