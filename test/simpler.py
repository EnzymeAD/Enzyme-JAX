from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import jax.numpy as jnp
import enzyme_ad.jax as enzyme_jax

def test(x, y, z, w):
  return (x + y) @ z

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(2, 2, 2, 2))
        b = jax.random.uniform(kb, shape=(2, 2, 2, 2))
        c = jax.random.uniform(kc, shape=(2, 2, 2, 2))
        d = jax.random.uniform(kd, shape=(2, 2, 2, 2))

        aa = jnp.array([[1, 2], [3, 4]])
        bb = jnp.array([[5, 6], [7, 8]])
        cc = jnp.array([[3, 4], [5, 6]])

        eres = efunc(a, b, c, d)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
