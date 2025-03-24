from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax


def test(a, b, c, d, e):
    def cond(x):
        return x[0] < 5
    
    def body(x):
        return (
            x[0] + 1, x[1], x[2], x[3], x[4]
        )
    
    init_state = (
        0,
        b, c, d, e
    )
    return jax.lax.while_loop(cond, body, init_state)

class Simple(absltest.TestCase):
    def test_simple_random(self):
        a = jax.random.uniform(jax.random.PRNGKey(0), shape=(3, 3)) * 0.1
        b = jax.random.uniform(jax.random.PRNGKey(1), shape=(3, 3)) * 0.1
        c = jax.random.uniform(jax.random.PRNGKey(2), shape=(3, 3)) * 0.1
        d = jax.random.uniform(jax.random.PRNGKey(3), shape=(3, 3)) * 0.1
        e = jax.random.uniform(jax.random.PRNGKey(4), shape=(3, 3)) * 0.1
        
        jres = test(a, b, c, d, e)
        eres = enzyme_jax.enzyme_jax_ir(
            pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass")
        )(test)(a, b, c, d, e)

        print(eres)
if __name__ == "__main__":
    absltest.main()
