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

from . import enzyme_call

LANG_CPP = enzyme_call.Language.CPP
LANG_LLVM = enzyme_call.Language.LLVM
LANG_MHLO = enzyme_call.Language.MHLO


def resource_dir():
    import os

    dn = os.path.dirname(enzyme_call.__file__)
    res = os.path.join(dn, "..", "..", "clang", "staging")
    return res


def cflags():
    import platform
    import os

    if platform.system() == "Darwin":
        res = (
            "-isysroot",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
            "-isystem",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1",
            "-internal-isystem",
            os.path.join(resource_dir(), "include"),
            "-internal-externc-isystem",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include",
            "-internal-externc-isystem",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include",
            "-fgnuc-version=4.2.1",
        )
    else:
        res = ()
        if os.getenv("ENABLE_GDBLISTENER") is not None:
            res = res + (
                "-debug-info-kind=standalone",
                "-dwarf-version=5",
                "-debugger-tuning=gdb",
            )
    return res


def _enzyme_primal_impl(
    *args_flat: jax.Array,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.Array]:
    del args_flat, source, out_shapes
    raise RuntimeError("must be JIT'ed")


def _enzyme_fwd_impl(
    *args_flat: jax.Array,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.Array]:
    del args_flat, source, out_shapes
    raise RuntimeError("must be JIT'ed")


def _enzyme_aug_impl(
    *args_flat: jax.Array,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.Array]:
    del args_flat, source, out_shapes
    raise RuntimeError("must be JIT'ed")


def _enzyme_shadow_aug_impl(
    *args_flat: jax.Array,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.Array]:
    del args_flat, source, out_shapes
    raise RuntimeError("must be JIT'ed")


def _enzyme_rev_impl(
    *args_flat: jax.Array,
    source,
    fn: str,
    argv: Sequence[str],
    in_shapes,
    lang: enzyme_call.Language,
) -> Sequence[jax.Array]:
    del args_flat, source, out_shapes
    raise RuntimeError("must be JIT'ed")


def _enzyme_primal_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.core.ShapedArray]:
    # TODO: we may attempt some lightweight parsing of source to extract the
    # result types instead.
    return out_shapes


def _enzyme_fwd_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.core.ShapedArray]:
    del source, fn, args_flat
    return tuple(o for o in out_shapes for _ in range(2))


def absmaketup(ty):
    tystr = ty.dtype.__str__()
    tystr = {"float32": "float", "float64": "double"}[tystr]
    return (tystr, ty.shape)


def _enzyme_aug_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.core.ShapedArray]:
    in_shapes = args_flat

    prev_out_shapes = out_shapes

    out_shapes = [absmaketup(a) for a in out_shapes]

    in_shapes = [absmaketup(a) for a in in_shapes]

    if lang == LANG_MHLO:
        (in_tree, func) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, args_flat)
        lowered_func = jax.jit(func).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="mhlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()

    tapeSize, tmpSize = enzyme_call.tape_and_tmp_size(
        source, fn, out_shapes, in_shapes, argv, lang
    )
    res = tuple(prev_out_shapes) + (
        jax.core.ShapedArray((tapeSize,), (jax.numpy.int8)),
    )
    return res


def _enzyme_shadow_aug_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[jax.core.ShapedArray]:
    return out_shapes


def _enzyme_rev_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    in_shapes,
    lang: enzyme_call.Language,
) -> Sequence[jax.core.ShapedArray]:
    return tuple(
        jax.core.ShapedArray(shape, dejaxify(tyid)) for (shape, tyid) in in_shapes
    )


def maketup(ty):
    ty = ir.RankedTensorType(ty)
    tystr = ty.element_type.__str__()
    tystr = {"f32": "float", "f64": "double", "i32": "int32_t", "i64": "int64_t"}[tystr]
    return (tystr, ty.shape)


def to_jax(ty):
    tystr = ty.__str__()
    return {"f32": jnp.float32, "f64": jnp.float64}[tystr]


def _enzyme_primal_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types))
    in_shapes = list(map(lambda x: maketup(x.type), args_flat))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, func) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in)
        lowered_func = jax.jit(func).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="mhlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source, fn, out_shapes, in_shapes, argv, enzyme_call.ABI.Primal, lang
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    mlir_args = (identifier_op,) + in_args

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        out_types = out_types + (sa,)

    custom_call = stablehlo.CustomCallOp(
        out_types, mlir_args, call_target_name="jaxzyme.primal"
    )

    results = custom_call.results

    if tmpBuf != 0:
        results = results[:-1]

    return results


def _enzyme_fwd_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types[::2]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, func) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in[::2])
        lowered_func = jax.jit(func).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="mhlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i // 2 in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source, fn, out_shapes, in_shapes, argv, enzyme_call.ABI.Forward, lang
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    mlir_args = (identifier_op,) + in_args

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        out_types = out_types + (sa, sa)

    custom_call = stablehlo.CustomCallOp(
        out_types, mlir_args, call_target_name="jaxzyme.fwd"
    )

    results = custom_call.results
    if tmpBuf != 0:
        results = results[:-2]

    return results


def _enzyme_aug_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types[: len(out_types) - 1]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, func) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in)
        lowered_func = jax.jit(func).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="mhlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source, fn, out_shapes, in_shapes, argv, enzyme_call.ABI.Augmented, lang
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        out_types = out_types + (sa,)

    mlir_args = (identifier_op,) + in_args
    custom_call = stablehlo.CustomCallOp(
        out_types, mlir_args, call_target_name="jaxzyme.aug"
    )

    results = custom_call.results
    if tmpBuf != 0:
        results = results[:-1]
    return results


def _enzyme_rev_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source,
    fn: str,
    argv: Sequence[str],
    in_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
) -> Sequence[ir.Value]:
    del in_shapes

    pre_in_types = tuple(
        itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
    )

    in_shapes = list(map(maketup, pre_in_types))
    pre_in_shapes = in_shapes

    out_shapes = list(map(lambda x: maketup(x.type), args_flat[1:]))

    in_args = (*args_flat,)

    rev_return_types = pre_in_types

    kept = None
    if lang == LANG_MHLO:
        (in_tree, func) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_out)
        lowered_func = jax.jit(func).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="mhlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        # in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]
        rev_return_types = tuple(
            retty for (i, retty) in enumerate(rev_return_types) if i in kept
        )

    argv = tuple(argv) + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source, fn, out_shapes, in_shapes, argv, enzyme_call.ABI.Reverse, lang
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    mlir_args = (identifier_op,) + in_args

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        rev_return_types = rev_return_types + (sa,)

    custom_call = stablehlo.CustomCallOp(
        rev_return_types, mlir_args, call_target_name="jaxzyme.rev"
    )
    results = custom_call.results
    if tmpBuf != 0:
        results = results[:-1]
    if kept != None:
        results = []
        cur_idx = 0
        for i, ty in enumerate(pre_in_types):
            if i in kept:
                results.append(custom_call.results[cur_idx])
                cur_idx += 1
            else:
                ty = ir.RankedTensorType(ty)
                shape = ty.shape
                element_type = ty.element_type
                import numpy as np

                results.append(
                    stablehlo.ConstantOp(
                        ir.DenseElementsAttr.get(
                            np.zeros(shape, dtype=to_jax(element_type))
                        )
                    ).results[0]
                )
    return results


def ffi_call(
    *args,
    out_shapes: Sequence[jax.core.ShapedArray],
    source,
    fn: str = "f",
    argv: tuple[str] = (),
    lang: int = LANG_CPP,
):
    return _enzyme_primal_p.bind(
        *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes, lang=lang
    )


def cpp_call(
    *args,
    out_shapes: Sequence[jax.core.ShapedArray],
    source: str,
    fn: str = "f",
    argv: tuple[str] = (),
):
    return ffi_call(
        *args, source=source, fn=fn, argv=argv, out_shapes=out_shapes, lang=LANG_CPP
    )


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


def enzyme_jvp(arg_primals, arg_tangents, **kwargs):
    # TODO propagate activity info rather than make_zero
    def make_zero(tan, prim):
        return lax.zeros_like_array(prim) if type(tan) is ad.Zero else tan

    arg_tangents = tuple(make_zero(t, p) for (t, p) in zip(arg_tangents, arg_primals))
    args = tuple(v for t in zip(arg_primals, arg_tangents) for v in t)
    shadconv = _enzyme_fwd_p.bind(
        *args,
        source=kwargs["source"],
        fn=kwargs["fn"],
        argv=kwargs["argv"],
        out_shapes=kwargs["out_shapes"],
        lang=kwargs["lang"],
    )
    res = (shadconv[0::2], shadconv[1::2])
    return res


ad.primitive_jvps[_enzyme_primal_p] = enzyme_jvp


def jaxify(x):
    return {"float32": 0, "float64": 1}[x.__str__()]


def dejaxify(x):
    return {0: jnp.float32, 1: jnp.float64}[x]


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

    outs_known = trace.default_process_primitive(_enzyme_aug_p, primals, kwargs)

    shadow_aug_args = (trace.full_raise(outs_known[-1]),) + primals + tangents
    shadows_known = trace.default_process_primitive(
        _enzyme_shadow_aug_p, shadow_aug_args, kwargs
    )

    outs = tuple(v for tup in zip(outs_known[:-1], shadows_known) for v in tup)
    return outs


pe.custom_partial_eval_rules[_enzyme_fwd_p] = fwd_partial_eval


def enzyme_vjp(shadow_rets, *prim_args, **kwargs):
    out_shapes = kwargs["out_shapes"]
    del kwargs["out_shapes"]
    shadows = [ad.is_undefined_primal(x) for x in prim_args]
    tape = prim_args[0]
    prim_args = prim_args[1 : 1 + (len(prim_args) - 1) // 2]
    prim_args = tuple(
        jnp.ones(x.aval.shape, x.aval.dtype) if ad.is_undefined_primal(x) else x
        for x in prim_args
    )
    in_shapes = tuple((a.shape, jaxify(a.dtype)) for a in prim_args)

    args = (tape,) + tuple(shadow_rets)
    shadconv = _enzyme_rev_p.bind(*args, **kwargs, in_shapes=in_shapes)
    res = (None,) + tuple(None for _ in range(len(shadconv))) + tuple(shadconv)
    return res


ad.primitive_transposes[_enzyme_shadow_aug_p] = enzyme_vjp


def enzyme_jax_ir(argv=()):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @jax.jit
        def wrapped(*args: Any):
            args_flat, in_tree = jax.tree_util.tree_flatten(args)
            out_shape = jax.eval_shape(func, *args)
            out_shape_flat, out_tree = jax.tree_util.tree_flatten(out_shape)
            out_shape_flat = [
                jax.core.ShapedArray(o.shape, o.dtype) for o in out_shape_flat
            ]
            out_flat = ffi_call(
                *args_flat,
                source=(in_tree, func),
                fn="",
                out_shapes=out_shape_flat,
                argv=argv,
                lang=LANG_MHLO,
            )
            return jax.tree_util.tree_unflatten(out_tree, out_flat)

        return wrapped

    return decorator
