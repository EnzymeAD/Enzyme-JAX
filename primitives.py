"""JAX primitives for Enzyme connection."""

from functools import partial
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


def _enzyme_primal_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_fwd_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_primal_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  del source, fn, args_flat

  # TODO: we may attempt some lightweight parsing of source to extract the
  # result types instead.
  return tuple(out_shapes)

def _enzyme_fwd_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  print("absfwd", args_flat, out_shapes)
  del source, fn, args_flat

  # each return is duplicated
  return tuple(o for o in out_shapes for _ in range(2))

def maketup(ty):
  ty = ir.RankedTensorType(ty)
  tystr = ty.element_type.__str__()
  tystr = {'f32':'float','f64':'double'}[tystr]
  return (tystr, ty.shape)


def _enzyme_primal_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  out_shapes = list(map(maketup, out_types))
  in_shapes = list(map(lambda x: maketup(x.type), args_flat))

  import os
  dn = os.path.dirname(enzyme_call.__file__)
  dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
  argv = [ "-resource-dir", dn]
  mode = 0
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      out_types, mlir_args, call_target_name="jaxzyme.primal"
  )

  return custom_call.results


def _enzyme_fwd_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  out_shapes = list(map(maketup, out_types[::2]))
  if out_types[::2] != out_types[1::2]:
    print(out_types)
  
  in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

  if tuple(map(lambda x:x.type, args_flat[::2])) != tuple(map(lambda x:x.type, args_flat[::2])):
    print(args_flat)

  import os
  dn = os.path.dirname(enzyme_call.__file__)
  dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
  argv = [ "-resource-dir", dn]
  mode = 1
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      out_types, mlir_args, call_target_name="jaxzyme.fwd"
  )

  return custom_call.results

@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def enzyme_primal(source: str, fn:str, out_shapes: Sequence[jax.core.ShapedArray], *args):
  print("args", args)
  return _enzyme_primal_p.bind(
      *args, source=source, fn=fn, out_shapes=out_shapes)

_enzyme_primal_p = jax.core.Primitive("enzyme_primal")
_enzyme_primal_p.multiple_results = True
_enzyme_primal_p.def_impl(_enzyme_primal_impl)
_enzyme_primal_p.def_abstract_eval(_enzyme_primal_abstract_eval)
jax_mlir.register_lowering(_enzyme_primal_p, _enzyme_primal_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.primal", enzyme_call.get_cpu_callback(), platform="cpu"
)

@enzyme_primal.defjvp
def enzyme_fwd(source: str, fn:str, out_shapes: Sequence[jax.core.ShapedArray], *args):
  print("fwdargs", args)
  args = (a[0] for a in args)
  print("fwdargs2", args)
  shadconv = _enzyme_fwd_p.bind(
      *args, source=source, fn=fn, out_shapes=out_shapes)
  return (shadconv[0::2], shadconv[1::2])

_enzyme_fwd_p = jax.core.Primitive("enzyme_fwd")
_enzyme_fwd_p.multiple_results = True
_enzyme_fwd_p.def_impl(_enzyme_fwd_impl)
_enzyme_fwd_p.def_abstract_eval(_enzyme_fwd_abstract_eval)
jax_mlir.register_lowering(_enzyme_fwd_p, _enzyme_fwd_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.fwd", enzyme_call.get_cpu_callback(), platform="cpu"
)

@jax.jit
def do_something(ones):
  print("ones", ones)
  shape = jax.core.ShapedArray(ones.shape, ones.dtype)
  a, b = enzyme_primal("""
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
  """, "myfn", [shape, shape], ones)
  c = enzyme_primal(
      """
template<typename T1, typename T2>
void myfn(T1& out0, const T2& in1) {
  out0 = 56.0f;
}
""", "myfn", [jax.core.ShapedArray([4, 4], jnp.float32)], a
  )
  return a, b, c


def main(args: Sequence[str]):
  del args
  ones = jnp.ones((2, 3), jnp.float32)
  x, y, z = do_something(ones)

  print(x)
  print(y)
  print(z)

  primals, tangents = jax.jvp(do_something, [ones], [ones])
  # primals, tangents = jax.jvp(do_something, (ones,), (ones,) )
  print(primals)
  print(tangents)


if __name__ == "__main__":
  main(sys.argv)
#  app.run(main)

