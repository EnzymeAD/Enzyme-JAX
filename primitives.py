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

def _enzyme_aug_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_rev_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    in_shapes
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
  del source, fn, args_flat

  # each return is duplicated
  return tuple(o for o in out_shapes for _ in range(2))

def _enzyme_aug_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  del source, fn, args_flat

  # each return is duplicated
  return tuple(out_shapes) + (jax.core.ShapedArray((80,), (jax.numpy.int8)),)

def _enzyme_rev_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    in_shapes
) -> Sequence[jax.core.ShapedArray]:
  del source, fn, args_flat

  return tuple(jax.core.ShapedArray(shape, dejaxify(tyid)) for (shape, tyid) in in_shapes)

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
  
  in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

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


def _enzyme_aug_lowering(
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

  out_shapes = list(map(maketup, out_types[:len(out_types)-1]))
  
  in_shapes = list(map(lambda x: maketup(x.type), args_flat))

  import os
  dn = os.path.dirname(enzyme_call.__file__)
  dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
  argv = [ "-resource-dir", dn]
  mode = 2
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      out_types, mlir_args, call_target_name="jaxzyme.aug"
  )
  print(custom_call)

  return custom_call.results


def _enzyme_rev_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source: str,
    fn: str,
    in_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del in_shapes

  in_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  in_shapes = list(map(maketup, in_types))

  out_shapes = list(map(lambda x: maketup(x.type), args_flat[1:]))  

  import os
  dn = os.path.dirname(enzyme_call.__file__)
  dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
  argv = [ "-resource-dir", dn]
  mode = 2
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      in_types, mlir_args, call_target_name="jaxzyme.rev"
  )

  return custom_call.results

# @partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def enzyme_primal(source: str, fn:str, out_shapes: Sequence[jax.core.ShapedArray], *args):
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

# @enzyme_primal.defjvp
def enzyme_fwd(source: str, fn:str, out_shapes: Sequence[jax.core.ShapedArray], *args):
  args = (a[0] for a in args)
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

def jaxify(x):
  return {'float32':0, 'float64':1}[x.__str__()]

def dejaxify(x):
  return {0:jnp.float32, 1:jnp.float64}[x]

def enzyme_aug(source: str, fn:str, out_shapes: Sequence[jax.core.ShapedArray], *args):
  shadconv = _enzyme_aug_p.bind(
      *args, source=source, fn=fn, out_shapes=out_shapes)
  print(" aug normconv", shadconv)
  shadconv = ( shadconv[0:len(shadconv)-1], (shadconv[-1],  tuple((a.shape, jaxify(a.dtype)) for a in args) ) )
  print("aug shadowconv", shadconv)
  return shadconv

_enzyme_aug_p = jax.core.Primitive("enzyme_aug")
_enzyme_aug_p.multiple_results = True
_enzyme_aug_p.def_impl(_enzyme_aug_impl)
_enzyme_aug_p.def_abstract_eval(_enzyme_aug_abstract_eval)
jax_mlir.register_lowering(_enzyme_aug_p, _enzyme_aug_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.aug", enzyme_call.get_cpu_callback(), platform="cpu"
)

def enzyme_rev(source: str, fn:str, out_shapes: Sequence[jax.core.ShapedArray], tape, args):
  (tape, in_shapes) = tape
  assert tape.dtype == jnp.int8
  assert tape.shape == (80,)
  args = (tape,) + tuple(args)
  shadconv = _enzyme_rev_p.bind(
      *args, source=source, fn=fn, in_shapes=in_shapes)
  return tuple(shadconv)

_enzyme_rev_p = jax.core.Primitive("enzyme_rev")
_enzyme_rev_p.multiple_results = True
_enzyme_rev_p.def_impl(_enzyme_rev_impl)
_enzyme_rev_p.def_abstract_eval(_enzyme_rev_abstract_eval)
jax_mlir.register_lowering(_enzyme_rev_p, _enzyme_rev_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.rev", enzyme_call.get_cpu_callback(), platform="cpu"
)

enzyme_primal.defvjp(enzyme_aug, enzyme_rev)

@jax.jit
def do_something(ones):
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
  )[0]
  return a, b, c


def main(args: Sequence[str]):
  del args
  ones = jnp.ones((2, 3), jnp.float32)
  # x, y, z = do_something(ones)
  x = ones
  y = ones
  z = jnp.ones((4, 4), jnp.float32)
  # z = [jnp.ones((4, 4), jnp.float32),]
  print(x)
  print(y)
  print(z)

  # primals, tangents = jax.jvp(do_something, [ones], [ones])
  # primals, tangents = jax.jvp(do_something, (ones,), (ones,) )
  # print(primals)
  # print(tangents)


  primals, f_vjp = jax.vjp(do_something, ones)
  (grads,) = f_vjp((x, y, z))
  # primals, tangents = jax.jvp(do_something, (ones,), (ones,) )
  print(primals)
  print(grads)


if __name__ == "__main__":
  main(sys.argv)
#  app.run(main)

