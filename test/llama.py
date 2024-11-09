from absl.testing import absltest

from test_utils import *


def rmsnorm(x, weight):
    import jax.numpy as jnp

    ss = 1 / jnp.sqrt(x.dot(x) / x.shape[0] + 1e-5)
    return weight * x * ss


def softmax(x):
    import jax.numpy as jnp

    max_val = jnp.max(x)
    x = jnp.exp(x - max_val)
    return x / sum(x)


def sigmoid(x):
    import jax.numpy as jnp

    return 1 / (1 + jnp.exp(-x))


def silu(x):
    return x * sigmoid(x)


# Token is token value
asserts = True


def forward(x, config, weights, key_cache, value_cache):
    import jax.numpy as jnp

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


class Llama(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp
        import jax.random

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
        self.count = 100 if jax.default_backend() == "cpu" else 1000
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends

        self.ins = [x, weights, key_cache, value_cache]
        self.dins = [dx, weights, key_cache, value_cache]
        self.douts = dx
        self.tol = 5e-5


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
