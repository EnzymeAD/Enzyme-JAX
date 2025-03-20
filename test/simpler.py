from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax


def test(a, b, c, d, e):
    def cond(x):
        return x[0] < 5
    
    def body(x):
        # tan(x) => sin(x)/cos(x)
        tan_val = jnp.tan(x[1])
        sin_cos_ratio = jnp.sin(x[1]) / jnp.cos(x[1])
        
        # sin(-x) => -sin(x) and cos(-x) => cos(x)
        sin_neg_x = jnp.sin(-x[2])
        cos_neg_x = jnp.cos(-x[2])
        
        # transpose distribution over trig functions
        sin_transpose = jnp.sin(x[1].T)
        sin_x_transpose = jnp.sin(x[1]).T
        
        # angle addition formulas
        sum_input = x[1] + x[2]
        sin_sum = jnp.sin(sum_input)
        sin_formula = jnp.sin(x[1]) * jnp.cos(x[2]) + jnp.cos(x[1]) * jnp.sin(x[2])
        
        cos_sum = jnp.cos(sum_input)
        cos_formula = jnp.cos(x[1]) * jnp.cos(x[2]) - jnp.sin(x[1]) * jnp.sin(x[2])

        result1 = sin_sum @ cos_sum
        result2 = sin_formula @ cos_formula
        return (
            x[0] + 1,
            tan_val @ sin_cos_ratio,
            sin_neg_x @ cos_neg_x,
            sin_transpose @ sin_x_transpose,
            result1 @ result2
        )
    
    init_state = (
        0,
        a + d,
        b @ c + e,
        a.T @ b,
        c @ d.T
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

        for i, (j, e) in enumerate(zip(jres, eres)):
            self.assertTrue(jnp.allclose(j, e, rtol=1e-3, atol=1e-3), 
                           f"Mismatch at result {i}: JAX={j}, Enzyme={e}")

if __name__ == "__main__":
    absltest.main()
