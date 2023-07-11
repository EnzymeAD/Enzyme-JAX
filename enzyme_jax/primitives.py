"""JAX primitives for Enzyme connection."""

from functools import partial
from collections.abc import Callable, Sequence
from typing import Any
import itertools
import sys

import jax
from jax import lax
from jax.interpreters import mlir as jax_mlir
from jax.interpreters import ad
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

def _enzyme_shadow_aug_impl(
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


def _enzyme_shadow_aug_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source: str,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
) -> Sequence[jax.core.ShapedArray]:
  return out_shapes

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

  argv = tuple(argv) + ( "-resource-dir", resource_dir()) + cflags()
  mode = 3
  identifier = enzyme_call.create_enzyme_cpu_kernel(source, fn, out_shapes, in_shapes, argv, mode)
  identifier_attr = jax_mlir.dense_int_elements([identifier])
  identifier_op = stablehlo.ConstantOp(identifier_attr)

  mlir_args = (identifier_op, *args_flat)
  custom_call = stablehlo.CustomCallOp(
      in_types, mlir_args, call_target_name="jaxzyme.rev"
  )
  return custom_call.results

def cpp_call(*args, out_shapes: Sequence[jax.core.ShapedArray], source: str, fn:str="f", argv: tuple[str]=()):
  return _enzyme_primal_p.bind(
      *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes)

_enzyme_primal_p = jax.core.Primitive("enzyme_primal")
_enzyme_primal_p.multiple_results = True
_enzyme_primal_p.def_impl(_enzyme_primal_impl)
_enzyme_primal_p.def_abstract_eval(_enzyme_primal_abstract_eval)
jax_mlir.register_lowering(_enzyme_primal_p, _enzyme_primal_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.primal", enzyme_call.get_cpu_callback(), platform="cpu"
)

_enzyme_fwd_p = jax.core.Primitive("enzyme_fwd")
_enzyme_fwd_p.multiple_results = True
_enzyme_fwd_p.def_impl(_enzyme_fwd_impl)
_enzyme_fwd_p.def_abstract_eval(_enzyme_fwd_abstract_eval)
jax_mlir.register_lowering(_enzyme_fwd_p, _enzyme_fwd_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.fwd", enzyme_call.get_cpu_callback(), platform="cpu"
)

def cpp_fwdcall(*args, out_shapes: Sequence[jax.core.ShapedArray], source: str, fn:str="f", argv: tuple[str]=()):
  return _enzyme_fwd_p.bind(
      *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes)

def enzyme_jvp(arg_primals, arg_tangents, **kwargs):
  
  # TODO propagate activity info rather than make_zero
  def make_zero(tan, prim):
    return lax.zeros_like_array(prim) if type(tan) is ad.Zero else tan 

  arg_tangents = tuple(make_zero(t, p) for (t, p) in zip(arg_tangents, arg_primals))
  args = tuple(v for t in zip(arg_primals, arg_tangents) for v in t)
  shadconv = cpp_fwdcall(
      *args, source=kwargs['source'], fn=kwargs['fn'], argv=kwargs['argv'], out_shapes=kwargs['out_shapes'])
  res = (shadconv[0::2], shadconv[1::2])
  return res

ad.primitive_jvps[_enzyme_primal_p] = enzyme_jvp

def jaxify(x):
  return {'float32':0, 'float64':1}[x.__str__()]

def dejaxify(x):
  return {0:jnp.float32, 1:jnp.float64}[x]

_enzyme_aug_p = jax.core.Primitive("enzyme_aug")
_enzyme_aug_p.multiple_results = True
_enzyme_aug_p.def_impl(_enzyme_aug_impl)
_enzyme_aug_p.def_abstract_eval(_enzyme_aug_abstract_eval)
jax_mlir.register_lowering(_enzyme_aug_p, _enzyme_aug_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.aug", enzyme_call.get_cpu_callback(), platform="cpu"
)

_enzyme_shadow_aug_p = jax.core.Primitive("enzyme_shadow_aug")
_enzyme_shadow_aug_p.multiple_results = True
_enzyme_shadow_aug_p.def_impl(_enzyme_shadow_aug_impl)
_enzyme_shadow_aug_p.def_abstract_eval(_enzyme_shadow_aug_abstract_eval)

_enzyme_rev_p = jax.core.Primitive("enzyme_rev")
_enzyme_rev_p.multiple_results = True
_enzyme_rev_p.def_impl(_enzyme_rev_impl)
_enzyme_rev_p.def_abstract_eval(_enzyme_rev_abstract_eval)
jax_mlir.register_lowering(_enzyme_rev_p, _enzyme_rev_lowering, platform="cpu")

xla_client.register_custom_call_target(
    "jaxzyme.rev", enzyme_call.get_cpu_callback(), platform="cpu"
)


from jax._src.interpreters import partial_eval as pe

def fwd_partial_eval(trace, *args, **kwargs):
  assert len(args) % 2 == 0
  nr_primals = len(args) // 2
  primals, tangents = args[0::2], args[1::2]
  all_primals_known = all(p.is_known() for p in primals)
  some_tangents_unknown = any(not t.is_known() for t in tangents)

  if not (all_primals_known and some_tangents_unknown):
    return trace.default_process_primitive(_enzyme_fwd_p, args, kwargs)

  outs_known = trace.default_process_primitive(
      _enzyme_aug_p, primals, kwargs)

  shadow_aug_args = (trace.full_raise(outs_known[-1]),) + primals + tangents
  shadows_known = trace.default_process_primitive(
      _enzyme_shadow_aug_p, shadow_aug_args,
        kwargs)

  outs = tuple(v for tup in zip(outs_known[:-1], shadows_known) for v in tup)
  return outs

pe.custom_partial_eval_rules[_enzyme_fwd_p] = fwd_partial_eval

def enzyme_vjp(shadow_rets, *prim_args, **kwargs):
  out_shapes = kwargs['out_shapes']
  del kwargs['out_shapes']
  shadows = [ad.is_undefined_primal(x) for x in prim_args]
  tape = prim_args[0]
  prim_args = prim_args[1:1+(len(prim_args)-1)//2]
  prim_args = tuple(jnp.ones(x.aval.shape, x.aval.dtype) if ad.is_undefined_primal(x) else x for x in prim_args)
  in_shapes = tuple((a.shape, jaxify(a.dtype)) for a in prim_args)

  args = (tape, ) + tuple(shadow_rets)
  shadconv = _enzyme_rev_p.bind(
      *args, **kwargs, in_shapes=in_shapes)
  res = (None,) + tuple(None for _ in range(len(shadconv))) + tuple(shadconv)
  return res

ad.primitive_transposes[_enzyme_shadow_aug_p] = enzyme_vjp
