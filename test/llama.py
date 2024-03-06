from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax
import numpy as np
import timeit

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")


def rmsnorm(x, weight):
    ss = 1 / jnp.sqrt(x.dot(x) / x.shape[0] + 1e-5)
    return x # weight * x * ss


def softmax(x):
    # max_val = jnp.max(x)
    # x = jnp.exp(x - max_val)
    return x # / sum(x)


def sigmoid(x):
    return 1 # / (1 + jnp.exp(-x))


def silu(x):
    return x # * sigmoid(x)


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

        rotM = jnp.array([[0.0, -1.0], [1., 0.]])
        toconv.append(rotM)
    toconv2 = toconv[: kv_dim // 2] + [jnp.eye(2)] * (dim // 2 - kv_dim // 2)

    toconv = jnp.array(toconv)
    toconv2 = jnp.array(toconv2)

    keys2 = []
    values2 = []
    k = wk[0, :, :] @ x
    
    k_tmp = jnp.reshape(k, (dim // 2, 2))

    # dim == head_size * n_heads

    # Batched gemv
    k = jnp.reshape(jnp.einsum("ijk,ik -> ij", toconv2, k_tmp), (dim,))

    key_cache_l = key_cache[0, :, :]
    h = jnp.reshape(k, (1, dim))
    key_cache_l = jnp.append(key_cache_l, h, axis=0)

    x = key_cache_l

    return x


class Llama(absltest.TestCase):
    def test_llama_random(self):
        config = {
            "dim": 2,
            "hidden_dim": 768,
            "n_layers": 1,
            # "n_heads": 6,
            "n_heads": 1,
            "n_kv_heads": 1,
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

        func = partial(forward, config)

        jfunc = jax.jit(func)

        efunc = enzyme_jax.enzyme_jax_ir(argv=argv, pipeline_options=pipeline)(func)

        eres = efunc(x, weights, key_cache, value_cache)
        print("Enzyme primal", eres)
        res = jfunc(x, weights, key_cache, value_cache)
        print("Jax primal", res)
        print(" max error", jnp.max(jnp.abs(eres - res)))
        assert (jnp.abs(eres - res) < 1e-3).all()

        number = 1000
        print(
            "Enzyme primal",
            timeit.Timer(
                "efunc(x, weights, key_cache, value_cache)",
                globals={
                    "efunc": efunc,
                    "x": x,
                    "weights": weights,
                    "key_cache": key_cache,
                    "value_cache": value_cache,
                },
            ).timeit(number),
        )
        print(
            "JaX primal",
            timeit.Timer(
                "jfunc(x, weights, key_cache, value_cache)",
                globals={
                    "jfunc": jfunc,
                    "x": x,
                    "weights": weights,
                    "key_cache": key_cache,
                    "value_cache": value_cache,
                },
            ).timeit(number),
        )
        # jfunc = jax.jit(partial(forward, config))
        # mlir = jax.jit(partial(forward, config)).lower(1, weights, key_cache, value_cache).compiler_ir(dialect="mhlo")

        @jax.jit
        def jfwd(x, dx, weights, dweights, kc, dkc, vc, dvc):
            return jax.jvp(jfunc, (x, weights, kc, vc), (x, weights, dkc, dvc))

        @jax.jit
        def efwd(x, dx, weights, dweights, kc, dkc, vc, dvc):
            return jax.jvp(efunc, (x, weights, kc, vc), (x, weights, dkc, dvc))

        eres = efwd(
            x, dx, weights, dweights, key_cache, key_cache, value_cache, value_cache
        )
        print("Enzyme fwd", eres)
        jres = jfwd(
            x, dx, weights, dweights, key_cache, key_cache, value_cache, value_cache
        )
        print("Jax fwd", jres)
        print(
            "Enzyme fwd",
            timeit.Timer(
                "efwd(x, dx, weights, dweights, key_cache, key_cache, value_cache, value_cache)",
                globals={
                    "efwd": efwd,
                    "x": x,
                    "dx": dx,
                    "weights": weights,
                    "dweights": dweights,
                    "key_cache": key_cache,
                    "value_cache": value_cache,
                },
            ).timeit(number),
        )
        print(
            "JaX fwd",
            timeit.Timer(
                "jfwd(x, dx, weights, dweights, key_cache, key_cache, value_cache, value_cache)",
                globals={
                    "jfwd": jfwd,
                    "x": x,
                    "dx": dx,
                    "weights": weights,
                    "dweights": dweights,
                    "key_cache": key_cache,
                    "value_cache": value_cache,
                },
            ).timeit(number),
        )

        @jax.jit
        def jrev(x, weights, kc, vc, dx, dkc, dvc):
            primals, f_vjp = jax.vjp(jfunc, x, weights, kc, vc)
            return f_vjp(dx)  # , dkc, dvc)

        @jax.jit
        def erev(x, weights, kc, vc, dx, dkc, dvc):
            primals, f_vjp = jax.vjp(efunc, x, weights, kc, vc)
            return f_vjp(dx)  # , dkc, dvc)

        eres = erev(x, weights, key_cache, value_cache, res, dkc, dvc)
        print("Enzyme rev", eres)
        jres = jrev(x, weights, key_cache, value_cache, res, dkc, dvc)
        print("Jax rev", jres)

        print(
            "Enzyme rev",
            timeit.Timer(
                "erev(x, weights, key_cache, value_cache, dx, dkc, dvc)",
                globals={
                    "erev": erev,
                    "x": x,
                    "weights": weights,
                    "key_cache": key_cache,
                    "value_cache": value_cache,
                    "dx": dx,
                    "dkc": dkc,
                    "dvc": dvc,
                },
            ).timeit(number),
        )
        print(
            "JaX rev",
            timeit.Timer(
                "jrev(x, weights, key_cache, value_cache, dx, dkc, dvc)",
                globals={
                    "jrev": jrev,
                    "x": x,
                    "weights": weights,
                    "key_cache": key_cache,
                    "value_cache": value_cache,
                    "dx": dx,
                    "dkc": dkc,
                    "dvc": dvc,
                },
            ).timeit(number),
        )


if __name__ == "__main__":
    absltest.main()
