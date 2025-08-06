from flax.linen import Module, compact, DenseGeneral
from flax.linen.attention import dot_product_attention_weights
from flax.linen.dtypes import Dtype, promote_dtype
from flax.linen.linear import PrecisionLike
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

'''
class getAttentionWeights(Module):
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

        # Return output and attention weights
        return attn_weights
'''

def get_attention_weights(
    inputs_q: jnp.ndarray,
    inputs_kv: jnp.ndarray,
    *,
    num_heads: int,
    dtype: Optional[jnp.dtype] = None,
    param_dtype: jnp.dtype = jnp.float32,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.0,
    deterministic: Optional[bool] = True,
    precision: PrecisionLike = None,
    kernel_init: Callable = lecun_normal(),
    bias_init: Callable = zeros,
    use_bias: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Function version of multi-head dot-product attention returning attention weights only."""

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]
    assert qkv_features % num_heads == 0, "Memory dimension must be divisible by number of heads."
    head_dim = qkv_features // num_heads

    # Dense projection layer constructor
    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        dtype=dtype,
        param_dtype=param_dtype,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        use_bias=use_bias,
        precision=precision
    )

    # Project inputs
    query_proj = dense(name="query")(inputs_q)
    key_proj = dense(name="key")(inputs_kv)
    value_proj = dense(name="value")(inputs_kv)

    # Promote dtypes
    query_proj, key_proj, value_proj = promote_dtype(query_proj, key_proj, value_proj, dtype=dtype)
    dtype = query_proj.dtype

    # Checks
    assert query_proj.ndim == key_proj.ndim == value_proj.ndim, "q, k, v must have same rank."
    assert query_proj.shape[:-3] == key_proj.shape[:-3] == value_proj.shape[:-3], "q, k, v batch dims must match."
    assert query_proj.shape[-2] == key_proj.shape[-2] == value_proj.shape[-2], "q, k, v num_heads must match."
    assert key_proj.shape[-3] == value_proj.shape[-3], "k, v lengths must match."

    # Compute attention weights
    attn_weights = dot_product_attention_weights(
        query_proj,
        key_proj,
        bias=None,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision
    )

    return attn_weights


def compute_attn_weights(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] | None = None,
    qk_attn_weights_einsum: Callable[..., Array] | None = None,
    attn_weights_value_einsum: Callable[..., Array] | None = None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of ``[batch...,
      q_length, num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    value: values to be used in attention with shape of ``[batch..., kv_length,
      num_heads, v_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see ``jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable
      via ``mutable=['intermediates']`` in order to have that collection
      returned. If ``module`` is None, the attention weights will not be sowed.
    force_fp32_for_softmax: bool, whether to force the softmax to be computed in
      fp32. This is useful for mixed-precision training where higher precision
      is desired for numerical stability.
    einsum_dot_general: the dot_general to use in `jnp.einsum`.
    qk_attn_weights_einsum: the einsum for computing the attention weights. When
      unspecified, the default `jnp.einsum` will be used. This argument is
      mutually exclusive with `precision` and `einsum_dot_general`.
    attn_weights_value_einsum: the einsum for computing the product of the
      attention weights and the values. When unspecified, the default
      `jnp.einsum` will be used. This argument is mutually exclusive with
      `precision` and `einsum_dot_general`.

  Returns:
    Output of shape ``[batch..., q_length, num_heads, v_depth_per_head]``.

  Raises:
    ValueError: if both `precision`/`einsum_dot_general` and
    `qk_attn_weights_einsum`/`attn_weights_value_einsum` are
      specified.
  """
  if (qk_attn_weights_einsum and not attn_weights_value_einsum) or (
      not qk_attn_weights_einsum and attn_weights_value_einsum
  ):
    raise ValueError(
        'qk_attn_weights_einsum and attn_weights_value_einsum must be specified'
        ' together.'
    )
  if (precision or einsum_dot_general) and (
      qk_attn_weights_einsum or attn_weights_value_einsum
  ):
    raise ValueError(
        'precision/einsum_dot_general and'
        ' qk_attn_weights_einsum/attn_weights_value_einsum are mutually'
        ' exclusive. Please specify only one of them.'
    )

  query, key, value = promote_dtype(query, key, value, dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
      query,
      key,
      bias,
      mask,
      broadcast_dropout,
      dropout_rng,
      dropout_rate,
      deterministic,
      dtype,
      precision,
      module,
      force_fp32_for_softmax,
      einsum_dot_general=einsum_dot_general,
      einsum=qk_attn_weights_einsum,
  )

  return attn_weights
