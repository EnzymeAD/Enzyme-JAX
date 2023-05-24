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
    out_shapes: Sequence[jax.core.ShapedArray],
    dump_ir: bool,
) -> Sequence[jax.Array]:
  del dump_ir, args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")


def _enzyme_fwd_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    out_shapes: Sequence[jax.core.ShapedArray],
    dump_ir: bool,
) -> Sequence[jax.core.ShapedArray]:
  del dump_ir, source, args_flat

  # TODO: we may attempt some lightweight parsing of source to extract the
  # result types instead.
  return tuple(out_shapes)


def _enzyme_fwd_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source: str,
    out_shapes: Sequence[jax.core.ShapedArray],
    dump_ir: bool,
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )
  out_shapes = list(map(lambda x: ir.RankedTensorType(x).shape, out_types))
  # TODO: also pass data types, currently assuming float

  identifier = enzyme_call.create_enzyme_cpu_kernel(source, out_shapes, dump_ir)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      out_types, mlir_args, call_target_name="jaxzyme.fwd"
  )

  if dump_ir:
    identifier_op.print()
    print()
    custom_call.print()
    print()

  return custom_call.results


def enzyme_fwd(*args, source: str, out_shapes: Sequence[jax.core.ShapedArray]):
  return _enzyme_fwd_p.bind(
      *args, source=source, out_shapes=out_shapes, dump_ir=True
  )

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
  a, b = enzyme_fwd(ones, source="foobar", out_shapes=[shape, shape])
  c = enzyme_fwd(
      a, source="quux", out_shapes=[jax.core.ShapedArray([4, 4], jnp.float32)]
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

