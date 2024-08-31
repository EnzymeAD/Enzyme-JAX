from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def test(x, y, z, w):
    return jnp.concat([x, y], axis=2) @ jnp.concat([z, w], axis=1)

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(100, 2,1000))
        b = jax.random.uniform(kb, shape=(100, 2,1000))
        c = jax.random.uniform(kc, shape=(100, 1000,2))
        d = jax.random.uniform(kd, shape=(100, 1000,2))

        eres = efunc(a, b, c, d)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
