"""JAX primitives for Enzyme connection."""

from collections.abc import Callable, Sequence
from typing import Any
import itertools
import sys

import jax
from jax.interpreters import mlir as jax_mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo
from jax.lib import xla_client

import jax.numpy as jnp

#from build import enzyme_call
import enzyme_call

# TODO: init LLVM.


def _enzyme_fwd_impl(
    *args_flat: jax.Array,
    source: str,
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")


def _enzyme_fwd_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  del source, args_flat

  # TODO: we may attempt some lightweight parsing of source to extract the
  # result types instead.
  return tuple(out_shapes)


def _enzyme_fwd_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )
  def maketup(ty):
    ty = ir.RankedTensorType(ty)
    tystr = ty.element_type.__str__()
    tystr = {'f32':'float','f64':'double'}[tystr]
    return (tystr, ty.shape)

  out_shapes = list(map(maketup, out_types))
  in_shapes = list(map(lambda x: maketup(x.type), args_flat))
  # in_shapes = out_shapes # list(map(lambda x: ir.RankedTensorType(x).shape, args_flat))
  # TODO: also pass data types, currently assuming float

  import os
  dn = os.path.dirname(enzyme_call.__file__)
  dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
  argv = ["-v", "-resource-dir", dn, "-O2"]
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, out_shapes, in_shapes, argv)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      out_types, mlir_args, call_target_name="jaxzyme.fwd"
  )

  if True:
    identifier_op.print()
    print()
    custom_call.print()
    print()

  return custom_call.results


def enzyme_fwd(*args, source: str, out_shapes: Sequence[jax.core.ShapedArray]):
  return _enzyme_fwd_p.bind(
      *args, source=source, out_shapes=out_shapes)

_enzyme_fwd_p = jax.core.Primitive("enzyme_fwd")
_enzyme_fwd_p.multiple_results = True
_enzyme_fwd_p.def_impl(_enzyme_fwd_impl)
_enzyme_fwd_p.def_abstract_eval(_enzyme_fwd_abstract_eval)
jax_mlir.register_lowering(_enzyme_fwd_p, _enzyme_fwd_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.fwd", enzyme_call.get_cpu_callback(), platform="cpu"
)


@jax.jit
def do_something():
  ones = jnp.ones((2, 3), jnp.float32)
  shape = jax.core.ShapedArray(ones.shape, ones.dtype)
  a, b = enzyme_fwd(ones, source="""
template<std::size_t N, std::size_t M>
void myfn(enzyme::tensor<float, N, M>& out0, enzyme::tensor<float, N, M>& out1, const enzyme::tensor<float, N, M>& in0) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<M; k++) {
        out0[j][k] = in0[j][k] + 42;
      }
    }
    for (int j=0; j<2; j++) {
      for (int k=0; k<3; k++) {
        out0[j][k] = in0[j][k] + 2 * 42;
      }
    }
}
  """, out_shapes=[shape, shape])
  c = enzyme_fwd(
      a, source="""
template<typename T1, typename T2>
void myfn(T1& out0, const T2& in1) {
  out0 = 56.0f;
}
""", out_shapes=[jax.core.ShapedArray([4, 4], jnp.float32)]
  )
  return a, b, c


def main(args: Sequence[str]):
  del args
  x, y, z = do_something()
  print(x)
  print(y)
  print(z)


if __name__ == "__main__":
  main(sys.argv)
#  app.run(main)

