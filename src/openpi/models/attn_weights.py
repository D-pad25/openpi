from flax.linen import Module, compact, DenseGeneral
from flax.linen.attention import dot_product_attention_weights
from flax.linen.dtypes import Dtype, PrecisionLike, promote_dtype
from flax.linen.initializers import zeros, lecun_normal
from flax.core.frozen_dict import FrozenDict
from jax import lax
import jax.numpy as jnp
import functools
from typing import Optional, Callable, Tuple, Union, Any

Array = Any
Shape = Tuple[int, ...]
PRNGKey = Any
default_kernel_init = lecun_normal()

class MultiHeadDotProductAttention(Module):
    """Multi-head dot-product attention with optional attention weight extraction."""
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    decode: bool = False

    @compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None) -> Tuple[Array, Array]:
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(DenseGeneral,
                                  axis=-1,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype,
                                  features=(self.num_heads, head_dim),
                                  kernel_init=self.kernel_init,
                                  bias_init=self.bias_init,
                                  use_bias=self.use_bias,
                                  precision=self.precision)

        # Project inputs
        query = dense(name='query')(inputs_q)
        key = dense(name='key')(inputs_kv)
        value = dense(name='value')(inputs_kv)

        if self.decode:
            is_initialized = self.has_variable('cache', 'cached_key')
            cached_key = self.variable('cache', 'cached_key',
                                       jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value',
                                         jnp.zeros, value.shape, value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(f'Autoregressive cache shape error, expected {expected_shape}, got {query.shape}')
                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value += 1
                mask = mask & jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length))

        dropout_rng = None
        if self.dropout_rate > 0.:
            m_deterministic = deterministic if deterministic is not None else self.deterministic
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True

        # Ensure all Q, K, V tensors have the same dtype (e.g., float32, bfloat16, etc.)
        query, key, value = promote_dtype(query, key, value, dtype=self.dtype)
        dtype = query.dtype

        # === Shape and consistency checks ===

        # All tensors should have the same rank (e.g., 4D: [batch, seq_len, num_heads, head_dim])
        assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
        
        # Ensure batch dimensions match (e.g., batch size, optional spatial dims)
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            'q, k, v batch dims must match.')
        
        # All should have the same number of heads (used for slicing and attention heads)
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            'q, k, v num_heads must match.')
        
        # Key and value sequences must match in length (used during weighted sum)
        assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
        # Get attention weights
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=None,
            mask=mask,
            broadcast_dropout=self.broadcast_dropout,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=m_deterministic,
            dtype=dtype,
            precision=self.precision
        )  # shape: [batch..., num_heads, query_len, key_len]

        # Multiply weights by value
        x = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value, precision=self.precision)

        # Combine heads
        # x = x.reshape(*x.shape[:-2], -1)

        # Final output projection
        out = DenseGeneral(features=features,
                           axis=-1,
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           precision=self.precision,
                           name='out')(x)

        # Return output and attention weights
        return out, attn_weights
