from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def test(x, y, z, w):

  # Define padding configurations
  padding_config = [(1, 2, 0), (2, 1, 0)]

  # Pad the array
  padded_arr = jax.lax.pad(x, padding_value=0., padding_config=padding_config)
  return padded_arr

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(2, 2))
        b = jax.random.uniform(kb, shape=(2, 2, 2, 2))
        c = jax.random.uniform(kc, shape=(2, 2, 2, 2))
        d = jax.random.uniform(kd, shape=(2, 2, 2, 2))

        eres = efunc(a, b, c, d)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
