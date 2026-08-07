from absl.testing import absltest

from test_utils import EnzymeJaxBenchmark


class JAXPIAdvection(EnzymeJaxBenchmark):
    def setUp(self):
        import jax.numpy as jnp
        from jax import random
        from jax.tree_util import tree_map
        from jaxpi.archs import Mlp

        model = Mlp(
            num_layers=2,
            hidden_dim=16,
            out_dim=1,
            activation="tanh",
            periodicity={
                "period": (1.0,),
                "axis": (1,),
                "trainable": (False,),
            },
            fourier_emb=None,
            reparam=None,
        )

        t = jnp.asarray(0.25, dtype=jnp.float32)
        x = jnp.asarray(1.0, dtype=jnp.float32)
        params = model.init(random.PRNGKey(0), jnp.stack((t, x)))

        def forward(params, t, x):
            return model.apply(params, jnp.stack((t, x)))[0]

        self.fn = forward
        self.name = "jaxpi_advection_mlp_2x16"
        self.repeat = 2

        self.ins = [params, t, x]
        self.dins = [
            tree_map(lambda value: jnp.full_like(value, 0.01), params),
            jnp.asarray(0.01, dtype=jnp.float32),
            jnp.asarray(0.01, dtype=jnp.float32),
        ]
        self.douts = jnp.asarray(1.0, dtype=jnp.float32)

        self.atol = 1e-5
        self.rtol = 1e-5


if __name__ == "__main__":
    absltest.main()
