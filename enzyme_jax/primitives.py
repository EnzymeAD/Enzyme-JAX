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
from enzyme_jax import enzyme_call

def resource_dir():
  import os
  dn = os.path.dirname(enzyme_call.__file__)
  return os.path.join(dn, "..", "clang", "staging")

def cflags():
    import platform
    import os
    if platform.system() == 'Darwin':
        return ('-isysroot', '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk', "-isystem", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1", "-internal-isystem", os.path.join(resource_dir(), "include"), "-internal-externc-isystem", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include", "-internal-externc-isystem", "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include", "-fgnuc-version=4.2.1")
    else:
        return ()

def _enzyme_primal_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_fwd_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_aug_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray]
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_rev_impl(
    *args_flat: jax.Array,
    source: str,
    fn: str,
    argv: Sequence[str],
    in_shapes
) -> Sequence[jax.Array]:
  del args_flat, source, out_shapes
  raise RuntimeError("must be JIT'ed")

def _enzyme_primal_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    argv: Sequence[str],
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
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  del source, fn, args_flat

  # each return is duplicated
  return tuple(o for o in out_shapes for _ in range(2))

def absmaketup(ty):
  tystr = ty.dtype.__str__()
  tystr = {'float32':'float','float64':'double'}[tystr]
  return (tystr, ty.shape)

def _enzyme_aug_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  
  in_shapes = args_flat

  prev_out_shapes = out_shapes

  out_shapes = [absmaketup(a) for a in out_shapes]

  in_shapes = [absmaketup(a) for a in in_shapes]

  argv = argv + ( "-resource-dir", resource_dir()) + cflags()

  tapeSize = enzyme_call.tape_size(source, fn, out_shapes, in_shapes, argv)
  res = tuple(prev_out_shapes) + (jax.core.ShapedArray((tapeSize,), (jax.numpy.int8)),)
  return res

def _enzyme_rev_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    argv: Sequence[str],
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
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  out_shapes = list(map(maketup, out_types))
  in_shapes = list(map(lambda x: maketup(x.type), args_flat))

  argv = argv + ( "-resource-dir", resource_dir() ) + cflags()
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
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  out_shapes = list(map(maketup, out_types[::2]))
  
  in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

  argv = argv + ( "-resource-dir", resource_dir() ) + cflags()
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
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del out_shapes

  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  out_shapes = list(map(maketup, out_types[:len(out_types)-1]))
  
  in_shapes = list(map(lambda x: maketup(x.type), args_flat))

  argv = argv + ( "-resource-dir", resource_dir()) + cflags()
  mode = 2
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      out_types, mlir_args, call_target_name="jaxzyme.aug"
  )

  return custom_call.results


def _enzyme_rev_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source: str,
    fn: str,
    argv: Sequence[str],
    in_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[ir.Value]:
  del in_shapes

  in_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )

  in_shapes = list(map(maketup, in_types))

  out_shapes = list(map(lambda x: maketup(x.type), args_flat[1:]))  

  argv = argv + ( "-resource-dir", resource_dir()) + cflags()
  mode = 3
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      in_types, mlir_args, call_target_name="jaxzyme.rev"
  )

  return custom_call.results

@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def cpp_fwd_internal(source: str, fn:str, argv: Sequence[str], out_shapes: Sequence[jax.core.ShapedArray], *args):
  return _enzyme_primal_p.bind(
      *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes)

def cpp_fwd(*args, out_shapes: Sequence[jax.core.ShapedArray], source: str, fn:str="f", argv: tuple[str]=()):
  return cpp_fwd_internal(source, fn, argv, out_shapes, *args)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def cpp_rev_internal(source: str, fn:str, argv: Sequence[str], out_shapes: Sequence[jax.core.ShapedArray], *args):
  return _enzyme_primal_p.bind(
      *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes)

def cpp_rev(*args, out_shapes: Sequence[jax.core.ShapedArray], source: str, fn:str="f", argv: tuple[str]=()):
  return cpp_rev_internal(source, fn, argv, out_shapes, *args)

_enzyme_primal_p = jax.core.Primitive("enzyme_primal")
_enzyme_primal_p.multiple_results = True
_enzyme_primal_p.def_impl(_enzyme_primal_impl)
_enzyme_primal_p.def_abstract_eval(_enzyme_primal_abstract_eval)
jax_mlir.register_lowering(_enzyme_primal_p, _enzyme_primal_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.primal", enzyme_call.get_cpu_callback(), platform="cpu"
)

@cpp_fwd_internal.defjvp
def enzyme_fwd(source: str, fn:str, argv: Sequence[str], out_shapes: Sequence[jax.core.ShapedArray], *args):
  args = (a[0] for a in args)
  shadconv = _enzyme_fwd_p.bind(
      *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes)
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

def enzyme_aug(source: str, fn:str, argv: Sequence[str], out_shapes: Sequence[jax.core.ShapedArray], *args):
  shadconv = _enzyme_aug_p.bind(
      *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes)
  shadconv = ( shadconv[0:len(shadconv)-1], (shadconv[-1],  tuple((a.shape, jaxify(a.dtype)) for a in args) ) )
  return shadconv

_enzyme_aug_p = jax.core.Primitive("enzyme_aug")
_enzyme_aug_p.multiple_results = True
_enzyme_aug_p.def_impl(_enzyme_aug_impl)
_enzyme_aug_p.def_abstract_eval(_enzyme_aug_abstract_eval)
jax_mlir.register_lowering(_enzyme_aug_p, _enzyme_aug_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.aug", enzyme_call.get_cpu_callback(), platform="cpu"
)

def enzyme_rev(source: str, fn:str, argv: Sequence[str], out_shapes: Sequence[jax.core.ShapedArray], tape, args):
  (tape, in_shapes) = tape
  args = (tape,) + tuple(args)
  shadconv = _enzyme_rev_p.bind(
      *args, source=source, fn=fn, argv=argv, in_shapes=in_shapes)
  return tuple(shadconv)

_enzyme_rev_p = jax.core.Primitive("enzyme_rev")
_enzyme_rev_p.multiple_results = True
_enzyme_rev_p.def_impl(_enzyme_rev_impl)
_enzyme_rev_p.def_abstract_eval(_enzyme_rev_abstract_eval)
jax_mlir.register_lowering(_enzyme_rev_p, _enzyme_rev_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.rev", enzyme_call.get_cpu_callback(), platform="cpu"
)

cpp_rev_internal.defvjp(enzyme_aug, enzyme_rev)
