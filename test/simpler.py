from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import jax.numpy as jnp
import enzyme_ad.jax as enzyme_jax


def test(a, b, c, d, e, f, g, h, i):
  incr = 1
  def cond(x):
    return x[0] < 10
  def body(x):
    def cond2(i):
      return i < x[0] * 2
    def body2(i):
      return i + 1
    new_x0 = jax.lax.while_loop(cond2, body2, x[0])
    return (new_x0, x[1] @ c, x[2] @ c)
  return jax.lax.while_loop(cond, body, (1, a, b))

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jax.random.uniform(ka, shape=(3, 10, 10))
        b = jax.random.uniform(kb, shape=(3, 10, 10))
        c = jax.random.uniform(kc, shape=(3, 10, 10))
        d = jax.random.uniform(kd, shape=(20, 500))
        e = jax.random.uniform(ka, shape=(20, 500))
        f = jax.random.uniform(ka, shape=(5, 5))
        g = 1
        h = 3
        i = jax.random.uniform(ka, shape=(300, 90))
        j = jax.random.uniform(ka, shape=(90, 10))
        eres = efunc(a, b, c, d, e, f, g, h, i)
        print("enzyme forward", eres)

if __name__ == "__main__":
    absltest.main()
