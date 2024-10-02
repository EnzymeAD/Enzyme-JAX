from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import jax.numpy as jnp
import enzyme_ad.jax as enzyme_jax

def test(a, b, c, d, e, f, g, h, i):
  return a @ b, b @ c, jnp.concat((d, e))

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(1000, 1000))
        b = jax.random.uniform(kb, shape=(1000, 1000))
        c = jax.random.uniform(kc, shape=(1000, 1000))
        d = jax.random.uniform(kd, shape=(20, 500))
        e = jax.random.uniform(ka, shape=(20, 500))
        f = jax.random.uniform(ka, shape=(5, 5))
        g = jax.random.uniform(ka, shape=(4, 80))
        h = jax.random.uniform(ka, shape=(80, 300))
        i = jax.random.uniform(ka, shape=(300, 90))
        j = jax.random.uniform(ka, shape=(90, 10))
        eres = efunc(a, b, c, d, e, f, g, h, i)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
