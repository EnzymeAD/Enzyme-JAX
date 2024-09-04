from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax
from enzyme_ad.jax import (
    enzyme_jax_ir,
    NewXLAPipeline,
    OldXLAPipeline,
    JaXPipeline,
    hlo_opts,
)
import numpy as np
import timeit
from test_utils import *

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")


def rmsnorm(x, weight):
    ss = 1 / jnp.sqrt(x.dot(x) / x.shape[0] + 1e-5)
    return weight * x * ss


def softmax(x):
    max_val = jnp.max(x)
    x = jnp.exp(x - max_val)
    return x / sum(x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def silu(x):
    return x * sigmoid(x)


# Token is token value
asserts = True

pipeline = enzyme_jax.NewXLAPipeline(mlirad=True)
pipeline = enzyme_jax.JaXPipeline()
# pipeline = enzyme_jax.NewXLAPipeline(mlirad=False)


def forward(x, config, weights, key_cache, value_cache):
    pos = key_cache.shape[1]
    assert pos == key_cache.shape[1]
    assert pos == value_cache.shape[1]

    n_layers = config["n_layers"]
    seq_len = config["seq_len"]
    n_heads = config["n_heads"]
    vocab_size = config["vocab_size"]

    # Total number of parameters of the recurrent state
    dim = config["dim"]

    n_kv_heads = config["n_kv_heads"]

    # number of hidden dimensions?
    hidden_dim = config["hidden_dim"]

    # Number of parameters per head
    head_size = dim // n_heads

    # Number of heads per kv
    kv_mul = n_heads // n_kv_heads

    # Number of parameters in a kv
    kv_dim = dim // n_heads * n_kv_heads

    wo = weights["wo"]
    if asserts:
        if wo.shape != (n_layers, dim, dim):
            print(
                wo.shape,
                weights,
                (
                    n_layers,
                    dim,
                    kv_dim,
                    kv_mul,
                    head_size,
                    hidden_dim,
                    n_kv_heads,
                    vocab_size,
                    n_heads,
                    seq_len,
                    n_layers,
                ),
            )
        assert wo.shape == (n_layers, dim, dim)
    rms_ffn_weight = weights["rms_ffn_weight"]
    if asserts:
        assert rms_ffn_weight.shape == (n_layers, dim)
    w1 = weights["w1"]
    if asserts:
        assert w1.shape == (n_layers, hidden_dim, dim)
    w3 = weights["w3"]
    if asserts:
        assert w3.shape == (n_layers, hidden_dim, dim)
    w2 = weights["w2"]
    if asserts:
        assert w2.shape == (n_layers, dim, hidden_dim)

    rms_att_weight = weights["rms_att_weight"]
    if asserts:
        assert rms_att_weight.shape == (n_layers, dim)

    rms_final_weight = weights["rms_final_weight"]
    if asserts:
        assert rms_final_weight.shape == (dim,)
    wcls = weights["wcls"]
    if asserts:
        assert wcls.shape == (vocab_size, dim)

    # token_embedding_table = weights['token_embedding_table']
    # if asserts: assert token_embedding_table.shape == (vocab_size, dim)

    # x = token_embedding_table[token, :]
    # if asserts: assert x.shape == (dim, )

    wq = weights["wq"]
    if asserts:
        assert wq.shape == (n_layers, dim, dim)

    wk = weights["wk"]
    if asserts:
        assert wk.shape == (n_layers, kv_dim, dim)

    wv = weights["wv"]
    if asserts:
        assert wv.shape == (n_layers, kv_dim, dim)

    toconv = []

    for i in range(0, dim, 2):
        freq = 1 / jnp.power(10000, (i % head_size) / head_size)
        val = pos * freq
        fcr = jnp.cos(val)
        fci = jnp.sin(val)

        rotM = jnp.array([[fcr, -fci], [fci, fcr]])
        toconv.append(rotM)
    toconv2 = toconv[: kv_dim // 2] + [jnp.eye(2)] * (dim // 2 - kv_dim // 2)

    toconv = jnp.array(toconv)
    toconv2 = jnp.array(toconv2)

    keys2 = []
    values2 = []
    for l in range(n_layers):
        xb = rmsnorm(x, rms_att_weight[l, :])
        if asserts:
            assert xb.shape == (dim,)

        q = wq[l, :, :] @ xb
        if asserts:
            assert q.shape == (dim,)

        k = wk[l, :, :] @ xb
        if asserts:
            assert q.shape == (kv_dim,)

        v = wv[l, :, :] @ xb
        if asserts:
            assert q.shape == (kv_dim,)

        q_tmp = jnp.reshape(q, (dim // 2, 2))
        k_tmp = jnp.reshape(k, (dim // 2, 2))

        # dim == head_size * n_heads

        # Batched gemv
        k = jnp.reshape(jnp.einsum("ijk,ik -> ij", toconv2, k_tmp), (dim,))
        q = jnp.reshape(jnp.einsum("ijk,ik -> ij", toconv, q_tmp), (dim,))

        key_cache_l = key_cache[l, :, :]
        key_cache_l = jnp.append(key_cache_l, jnp.reshape(k, (1, dim)), axis=0)
        value_cache_l = value_cache[l, :, :]
        value_cache_l = jnp.append(value_cache_l, jnp.reshape(v, (1, dim)), axis=0)
        keys2.append(key_cache_l)
        values2.append(value_cache_l)

        xbs2 = []
        for h in range(n_heads):
            q2 = q[head_size * h : head_size * (h + 1)]
            if asserts:
                assert q2.shape == (head_size,)

            # For kv_mul consecutive heads, they share the same kv cache
            # reshape key_cache last dim from (kv_dim,) to (kv_mul, head_size)
            # generalized einsum reducing the last dim, the rest are batch
            att = []

            key_index = h // kv_mul

            att = jnp.einsum(
                "ij,j->i",
                key_cache_l[:, key_index * head_size : (key_index + 1) * head_size],
                q2,
            )

            att = att / jnp.sqrt(head_size)

            att = softmax(att)

            x_tmp = jnp.einsum(
                "ij,i->j",
                value_cache_l[:, key_index * head_size : (key_index + 1) * head_size],
                att,
            )

            xbs2.append(x_tmp)

        # Todo right concat
        xb = jnp.concatenate(xbs2, axis=None)

        xb2 = wo[l, :, :] @ xb
        if asserts:
            assert xb2.shape == (dim,)

        x += xb2

        # Rmsnorm and feedforward swiglu

        xb = rmsnorm(x, rms_ffn_weight[l, :])

        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)

        hb = w1[l, :, :] @ xb
        hb2 = w3[l, :, :] @ xb

        hb = silu(hb)

        hb = hb * hb2

        xb = w2[l, :, :] @ hb

        x += xb

    x = rmsnorm(x, rms_final_weight)
    logits = wcls @ x

    return x


partialopt = (
    "inline{default-pipeline=canonicalize max-iterations=4},"
    + """canonicalize,cse,
enzyme-hlo-generate-td{
            patterns=compare_op_canon<16>;
transpose_transpose<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;

cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;

concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
reshape_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;

bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
slice_reshape<1>;

dot_reshape_pad<1>;
pad_dot_general<1>(1);
broadcast_reduce<1>;
            },
            transform-interpreter,
            enzyme-hlo-remove-transform,cse"""
)

pipelines = [
    ("JaX  ", None, CurBackends),
    ("JaXPipe", JaXPipeline(), CurBackends),
    (
        "HLOOpt",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse"
        ),
        CurBackends,
    ),
    ("PartOpt", JaXPipeline(partialopt), CurBackends),
    ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
]


class Llama(EnzymeJaxTest):
    def setUp(self):
        config = {
            "dim": 288,
            "hidden_dim": 768,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 6,
            "vocab_size": 32000,
            "seq_len": 256,
        }

        n_layers = config["n_layers"]
        seq_len = config["seq_len"]
        n_heads = config["n_heads"]
        dim = config["dim"]
        n_kv_heads = config["n_kv_heads"]
        vocab_size = config["vocab_size"]
        hidden_dim = config["hidden_dim"]
        kv_dim = dim // n_heads * n_kv_heads
        head_size = dim // n_heads

        key = jax.random.PRNGKey(0)
        weights = {}
        dweights = {}

        for name, shape in [
            ("rms_att_weight", (n_layers, dim)),
            ("wq", (n_layers, dim, n_heads * head_size)),
            ("wk", (n_layers, dim, n_kv_heads * head_size)),
            ("wv", (n_layers, dim, n_kv_heads * head_size)),
            ("wo", (n_layers, dim, dim)),
            ("rms_ffn_weight", (n_layers, dim)),
            ("w1", (n_layers, hidden_dim, dim)),
            ("w2", (n_layers, dim, hidden_dim)),
            ("w3", (n_layers, hidden_dim, dim)),
            ("rms_final_weight", (dim,)),
            ("wcls", (vocab_size, dim)),
        ]:
            key, subkey = jax.random.split(key)
            key, subkey2 = jax.random.split(key)
            weights[name] = jax.random.uniform(subkey, shape=shape)
            dweights[name] = jax.random.uniform(subkey2, shape=shape)

        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=(dim,))
        key, subkey = jax.random.split(key)
        dx = jax.random.uniform(subkey, shape=(dim,))

        def partial(func, config):
            def sfn(x, weights, key_cache, value_cache):
                return func(x, config, weights, key_cache, value_cache)

            return sfn

        pos = 1
        key_cache = jnp.zeros((n_layers, pos, kv_dim))
        value_cache = jnp.zeros((n_layers, pos, kv_dim))

        key, subkey = jax.random.split(key)
        dkc = jax.random.uniform(subkey, shape=(n_layers, pos + 1, kv_dim))
        key, subkey = jax.random.split(key)
        dvc = jax.random.uniform(subkey, shape=(n_layers, pos + 1, kv_dim))

        self.fn = partial(forward, config)
        self.name = "llama"
        self.count = 1 # 100 if jax.default_backend() == "cpu" else 1000
        self.revprimal = False
        self.AllPipelines = pipelines
        self.AllBackends = CurBackends

        self.ins = [x, weights, key_cache, value_cache]
        self.dins = [dx, weights, key_cache, value_cache]
        self.douts = [dx]
        self.tol = 5e-5


if __name__ == "__main__":
    absltest.main()
