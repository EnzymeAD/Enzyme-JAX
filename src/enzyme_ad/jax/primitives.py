"""JAX primitives for Enzyme connection."""

from functools import partial
from collections.abc import Callable, Sequence
from typing import Any
import inspect
import itertools
import os
import tempfile
from absl import logging

import jax
from jax import lax
from jax.interpreters import mlir as jax_mlir
from jax.interpreters import ad
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo, func
import jax.numpy as jnp
import jax.extend
from jax._src.interpreters import partial_eval as pe

from . import enzyme_call

from .utils import default_nowheel_resource, default_linux_cflags

LANG_CPP = enzyme_call.Language.CPP
LANG_LLVM = enzyme_call.Language.LLVM
LANG_MHLO = enzyme_call.Language.MHLO

Primitive = jax.extend.core.Primitive

try:
    register_custom_call_target = partial(jax.ffi.register_ffi_target, api_version=0)
except AttributeError:
    from jax.lib import xla_client

    register_custom_call_target = xla_client.register_custom_call_target

if hasattr(enzyme_call, "register_enzymexla_xla_ffi"):
    enzyme_call.register_enzymexla_xla_ffi()


_aval_to_ir_types_impl: Any = jax_mlir.aval_to_ir_types
_aval_to_ir_types_takes_module_context = (
    len(inspect.signature(_aval_to_ir_types_impl).parameters) > 1
)


def _aval_to_ir_types(ctx: jax_mlir.LoweringRuleContext, aval):
    """Call aval_to_ir_types with its JAX-version-dependent signature."""
    if _aval_to_ir_types_takes_module_context:
        return _aval_to_ir_types_impl(ctx.module_context, aval)
    return _aval_to_ir_types_impl(aval)


class PipelineConfig:
    # Whether to use the new xla runtime
    def xla_runtime(self):
        raise NotImplementedError()

    # Pass pipeline of new runtime
    def pass_pipeline(self):
        raise NotImplementedError()

    # MLIR pass pipeline
    def mlir_ad(self):
        raise NotImplementedError()

    # Whether to re-inject into stablehloMLIR pass pipeline
    def stablehlo_inject(self):
        raise NotImplementedError()

    # Number of levels deep of AD
    def ad_level(self):
        raise NotImplementedError()

    def export_llvm(self):
        raise NotImplementedError()


class XLAPipeline:
    def __init__(self, name=None):
        self.exportname = name

    def xla_runtime(self):
        return False

    def pass_pipeline(self):
        return ""

    def mlir_ad(self):
        return False

    def stablehlo_inject(self):
        return False

    def export_llvm(self):
        return self.exportname


class JaXPipeline:
    def __init__(self, passes=""):
        self.passes = passes

    def pass_pipeline(self):
        return self.passes

    def mlir_ad(self):
        return True

    def stablehlo_inject(self):
        return True

    def ad_level(self):
        return self.passes.count("enzyme-wrap")


def optimization_passes(
    *,
    inline: bool = True,
    no_nan: bool = False,
    all_finite: bool = False,
    transpose_propagate: str = "up",
    reshape_propagate: str = "up",
    max_constant_threshold: int = 1024,
    enable_licm_optimization_passes: bool = True,
    loop_unswitch_threshold: int = 10,
    excluded_passes: list = [],
    enable_scatter_gather_optimization_passes: bool = True,
    enable_pad_optimization_passes: bool = True,
    enable_self_to_convolution_like_passes: bool = False,
    enable_structured_tensors_detection_passes: bool = False,
    enable_structured_tensors_passes: bool = True,
    enable_slice_to_batch_passes: bool = False,  # this are somewhat expensive to run
    enable_reduce_slice_fusion_passes: bool = True,
    enable_concat_to_batch_passes: bool = True,
    enable_loop_raising_passes: bool = True,
    enable_dot_formation_passes: bool = True,
    max_reused_const_prop_size: int = 1024,
    aggressive_propagation: bool = True,
):
    # Map propagation direction strings to C enum values
    propagate_map = {"none": 0, "up": 1, "down": 2}
    reshape_prop = propagate_map.get(reshape_propagate, 0)
    transpose_prop = propagate_map.get(transpose_propagate, 0)

    # Call the C API to get the transform passes list
    main_passes_str, _lower_passes_str = enzyme_call.get_transform_passes_list(
        max_constant_threshold,  # max_constant_threshold
        4,  # while_unroll_threshold
        reshape_prop,  # reshape_propagate
        transpose_prop,  # transpose_propagate
        no_nan,  # no_nan
        all_finite,  # all_finite
        False,  # dus_to_concat
        True,  # dus_slice_simplify
        False,  # sum_to_reducewindow
        False,  # sum_to_conv
        False,  # aggressive_sum_to_conv
        False,  # while_concat
        aggressive_propagation,  # aggressive_propagation
        False,  # is_sharded
        False,  # raise_shlo_to_blas_lapack
        False,  # recognize_comms
        False,  # lower_comms
        enable_structured_tensors_passes,  # enable_structured_tensors_detection_passes
        enable_structured_tensors_passes,  # enable_structured_tensors_passes
        enable_scatter_gather_optimization_passes,  # enable_scatter_gather_optimization_passes
        enable_slice_to_batch_passes,  # enable_slice_to_batch_passes
        enable_reduce_slice_fusion_passes,  # enable_reduce_slice_fusion_passes
        enable_concat_to_batch_passes,  # enable_concat_to_batch_passes
        enable_loop_raising_passes,  # enable_loop_raising_passes
        enable_licm_optimization_passes,  # enable_licm_optimization_passes
        loop_unswitch_threshold,  # loop_unswitch_threshold
        enable_pad_optimization_passes,  # enable_pad_optimization_passes
        enable_self_to_convolution_like_passes,  # enable_self_to_convolution_like_passes
        enable_dot_formation_passes,  # enable_dot_formation_passes
        max_reused_const_prop_size,  # max_reused_const_prop_size
        excluded_passes,  # excluded_passes
    )

    transform_passes = ",".join(
        [
            "enzyme-hlo-generate-td{patterns=" + main_passes_str + "}",
            "transform-interpreter",
            "enzyme-hlo-remove-transform",
        ]
    )

    func_passes = ",".join(["canonicalize", "cse", "canonicalize", transform_passes])

    if inline:
        func_passes = (
            "inline{default-pipeline=canonicalize max-iterations=4}" + "," + func_passes
        )
    return func_passes


def full_optimization_pass_pipeline(**kwargs):
    opt_passes = optimization_passes(**kwargs)

    enzyme_pass = 'enzyme{postpasses="arith-raise{stablehlo=true},enzyme-batch-to-stablehlo,canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize"}'

    propagate_down_passes = ""
    if (
        kwargs.get("transpose_propagate", "up") == "up"
        or kwargs.get("reshape_propagate", "up") == "up"
    ):
        propagate_down_passes = optimization_passes(
            **kwargs, transpose_propagate="down", reshape_propagate="down"
        )

    return ",".join(
        [
            "mark-func-memory-effects",
            opt_passes,
            "enzyme-batch",
            opt_passes,
            enzyme_pass,
            opt_passes,
            "canonicalize",
            "remove-unnecessary-enzyme-ops",
            "enzyme-simplify-math",
            opt_passes,
            propagate_down_passes,
        ]
    )


DefaultCPPPipeline = XLAPipeline()
DefaultJaXPipeline = JaXPipeline(full_optimization_pass_pipeline())


def pass_pipeline(options):
    if isinstance(options, str):
        return options
    else:
        return


def resource_dir():
    import os

    dn = os.path.dirname(enzyme_call.__file__)
    if os.getenv("ENZYME_BAZEL_NOWHEEL", None) is None:
        res = default_nowheel_resource(dn)
        os.path.join(
            dn, "..", "..", "..", "external", "llvm-project", "clang", "staging"
        )
    else:
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
        res = default_linux_cflags()
        if os.getenv("ENABLE_GDBLISTENER") is not None:
            res = res + (
                "-debug-info-kind=standalone",
                "-dwarf-version=5",
                "-debugger-tuning=gdb",
            )
    return res


def optimize_module(mod, pipeline=None):
    if pipeline is None:
        pipeline = full_optimization_pass_pipeline()
    enzyme_call.optimize_module(mod, pipeline)
    return


def _enzyme_primal_impl(
    *args_flat: jax.Array,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
    pipeline_options,
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
    pipeline_options,
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
    pipeline_options,
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
    pipeline_options,
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
    pipeline_options,
) -> Sequence[jax.Array]:
    del args_flat, source, in_shapes
    raise RuntimeError("must be JIT'ed")


def _enzyme_primal_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
    pipeline_options,
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
    pipeline_options,
) -> Sequence[jax.core.ShapedArray]:
    del source, fn, args_flat
    return tuple(o for o in out_shapes for _ in range(2))


def absmaketup(ty):
    tystr = ty.dtype.__str__()
    tystr = {"float32": "float", "float64": "double", "int32": "int32_t"}[tystr]
    return (tystr, ty.shape)


def lower(fn, vals, parameters=None, kwargs={}):
    if hasattr(fn, "trace"):
        return fn.trace(*vals, **kwargs).lower(_private_parameters=parameters)
    else:
        if parameters is not None:
            return fn.lower(*vals, _experimental_lowering_parameters=parameters)
        else:
            return fn.lower(*vals)


def _enzyme_aug_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
    pipeline_options,
) -> Sequence[jax.core.ShapedArray]:
    in_shapes = args_flat

    prev_out_shapes = out_shapes

    out_shapes = [absmaketup(a) for a in out_shapes]

    in_shapes = [absmaketup(a) for a in in_shapes]

    if lang == LANG_MHLO:
        in_tree, _, _, mfunc, jit_options = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        avals_in, avals_inkw = jax.tree_util.tree_unflatten(in_tree, args_flat)
        jit_options = dict(jit_options)
        lowered_func = lower(jax.jit(mfunc, **jit_options), avals_in, kwargs=avals_inkw)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = mhlo.operation.get_asm(enable_debug_info=True)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()

    tapeSize, tmpSize = enzyme_call.tape_and_tmp_size(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
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
    pipeline_options,
) -> Sequence[jax.core.ShapedArray]:
    return out_shapes


def _enzyme_rev_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    in_shapes,
    lang: enzyme_call.Language,
    pipeline_options,
) -> Sequence[jax.core.ShapedArray]:
    return tuple(
        jax.core.ShapedArray(shape, dejaxify(tyid)) for (shape, tyid) in in_shapes
    )


def maketup(ty):
    ty = ir.RankedTensorType(ty)
    tystr = ty.element_type.__str__()
    tystr = {
        "i1": "bool",
        "i8": "char",
        "bf16": "bfloat16",
        "f32": "float",
        "f64": "double",
        "i32": "int32_t",
        "i64": "int64_t",
        "ui32": "uint32_t",
        "ui64": "uint64_t",
        "complex<f32>": "std::complex<float>",
        "complex<f64>": "std::complex<double>",
    }[tystr]
    return (tystr, ty.shape)


def make_mlir_zero(ty):
    from jax._src.interpreters import mlir

    if isinstance(ty, mlir.ir.RankedTensorType):
        elty = ty.element_type
    else:
        elty = jax_mlir.dtype_to_ir_type(ty).element_type
    elem = (
        ir.IntegerAttr.get(elty, 0)
        if isinstance(elty, ir.IntegerType)
        else ir.FloatAttr.get(elty, 0.0)
    )
    return stablehlo.ConstantOp(ir.DenseElementsAttr.get_splat(ty, elem)).results[0]


def arg_activity_from_pipeline(pass_pipeline):
    start = pass_pipeline.index("argTys=")
    end = pass_pipeline.index(" ", start)
    acts = pass_pipeline[start + len("argTys=") : end].split(",")
    pre_act = pass_pipeline[: start + len("argTys=")]
    post_act = pass_pipeline[end:]
    return pre_act, acts, post_act


def ret_activity_from_pipeline(pass_pipeline):
    start = pass_pipeline.index("retTys=")
    end = pass_pipeline.index(" ", start)
    acts = pass_pipeline[start + len("retTys=") : end].split(",")
    pre_act = pass_pipeline[: start + len("retTys=")]
    post_act = pass_pipeline[end:]
    return pre_act, acts, post_act


def _dump_mlir_to_file(source, pass_pipeline):
    # bazel will zip up the outputs in this directory
    dump_mlir_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", None)
    if dump_mlir_dir is None:
        dump_mlir_dir = tempfile.gettempdir()

    tmpfile = tempfile.NamedTemporaryFile(
        suffix=".mlir", dir=dump_mlir_dir, delete=False
    )
    with open(tmpfile.name, "w") as f:
        f.write("// " + pass_pipeline + "\n")
        f.write(str(source))

    return tmpfile.name


def _enzyme_primal_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args_flat: ir.Value,
    source,
    fn: str,
    argv: Sequence[str],
    out_shapes: Sequence[jax.core.ShapedArray],
    lang: enzyme_call.Language,
    pipeline_options,
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(
        itertools.chain(*map(lambda x: _aval_to_ir_types(ctx, x), ctx.avals_out))
    )

    out_shapes = list(map(maketup, out_types))
    in_shapes = list(map(lambda x: maketup(x.type), args_flat))
    in_types = list(map(lambda x: x.type, args_flat))

    in_args = (*args_flat,)

    pass_pipeline = pipeline_options.pass_pipeline()

    out_idx_map = {i: -1 for i in range(len(out_shapes))}

    argv = argv + ("-resource-dir", resource_dir()) + cflags()

    if lang == LANG_MHLO:
        in_tree, in_idx_map, out_idx_map, mfunc, jit_options = source
        in_idx_map = dict(in_idx_map)
        out_idx_map = dict(out_idx_map)
        jit_options = dict(jit_options)
        print_mlir = False
        if "print_mlir" in jit_options:
            print_mlir = jit_options["print_mlir"]
            del jit_options["print_mlir"]
        assert len(out_idx_map) == len(out_shapes)

        orig_shapes = []
        orig_types = []
        seen = {}
        for i, shape in enumerate(in_shapes):
            if i not in in_idx_map:
                continue
            if in_idx_map[i] in seen:
                continue
            seen[in_idx_map[i]] = i
            orig_shapes.append(shape)
            orig_types.append(in_types[i])
        avals = [ctx.avals_in[seen[i]] for i in seen]
        if isinstance(mfunc, str):
            avals_in = avals
            kept = [i for (i, v) in enumerate(orig_shapes)]
            source = mfunc
        else:
            avals_in, avals_inkw = jax.tree_util.tree_unflatten(in_tree, avals)
            lowered_func = lower(
                jax.jit(mfunc, **jit_options),
                avals_in,
                ctx.module_context.lowering_parameters,
                kwargs=avals_inkw,
            )
            mhlo = lowered_func.compiler_ir(dialect="stablehlo")
            source = mhlo.operation.get_asm(enable_debug_info=True)
            kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(
            arg
            for (i, arg) in enumerate(in_args)
            if i not in in_idx_map or in_idx_map[i] in kept
        )
        if len(kept) != len(orig_shapes):
            if "argTys=" in pass_pipeline:
                pre_act, acts, post_act = arg_activity_from_pipeline(pass_pipeline)
                acts2 = [act for (i, act) in enumerate(acts) if i in kept]
                pass_pipeline = pre_act + ",".join(acts2) + post_act

            out_types = [
                shape
                for i, shape in enumerate(out_types)
                if out_idx_map[i] < 0 or out_idx_map[i] in kept
            ]
            out_shapes = list(map(maketup, out_types))

        # in_shapes = [shape for (i, shape) in enumerate(orig_shapes) if i in kept]
        in_shapes = [
            shape
            for (i, shape) in enumerate(in_shapes)
            if i not in in_idx_map or in_idx_map[i] in kept
        ]
        if pipeline_options.stablehlo_inject():
            mod = ctx.module_context.module.operation
            fns = []
            for f in mod.regions[0].blocks[0]:
                fns.append(f.sym_name.value)

            if len(pass_pipeline) > 0:
                pass_pipeline = (
                    pass_pipeline + ",tensor-empty-raise,drop-unsupported-attributes"
                )

            try:
                name, nmod = enzyme_call.run_pass_pipeline(fns, source, pass_pipeline)
            except Exception as e:
                filename = _dump_mlir_to_file(source, pass_pipeline)
                logging.exception("Enzyme MLIR dumped to %s", filename)
                raise e

            if print_mlir:
                if not isinstance(print_mlir, bool):
                    print_mlir.write(nmod)
                else:
                    print(str(nmod), flush=True)

            nmod = ir.Module.parse(nmod)
            fn = None
            pushtop = []
            for f in nmod.body:
                if "top_k_gt" in f.sym_name.value:
                    pushtop.append(f)
                mod.regions[0].blocks[0].append(f)
                if f.sym_name.value == name:
                    fn = f
            if fn is None:
                raise AssertionError(
                    "Could not find function named "
                    + name
                    + " in post opt module "
                    + str(nmod)
                    + ", pre opt module was "
                    + str(source)
                    + ' pipeline was "'
                    + pass_pipeline
                    + '"'
                )
            for f in pushtop[::-1]:
                f.move_before(next(mod.regions[0].blocks[0].__iter__()))
            if True:
                identifier_attr = jax_mlir.dense_int_elements([0])
                placeholderop = stablehlo.ConstantOp(identifier_attr)
                for op in list(fn.regions[0].blocks[0].operations)[:-1]:
                    op.move_before(placeholderop)
                for ba, arg in zip(fn.regions[0].blocks[0].arguments, in_args):
                    ba.replace_all_uses_with(arg)
                results = list(fn.regions[0].blocks[0].operations[0].operands)
                fn.regions[0].blocks[0].operations[0].erase()
                fn.erase()
                placeholderop.erase()
            else:
                callop = func.CallOp(fn, list(in_args))
                results = callop.results
            if len(results) != len(out_shapes):
                print(source)
                print(pass_pipeline)
                print(str(nmod))
                print(out_shapes, "\n", results, "\n", nmod)
            assert len(results) == len(out_shapes)
        else:
            assert len(ctx.module_context.platforms) == 1
            identifier, tmpBuf = enzyme_call.create_enzyme_kernel(
                source,
                fn,
                out_shapes,
                in_shapes,
                argv,
                enzyme_call.ABI.Primal,
                lang,
                pipeline_options.xla_runtime(),
                pass_pipeline,
                ctx.module_context.platforms[0],
            )
            identifier_attr = jax_mlir.dense_int_elements([identifier])
            identifier_op = stablehlo.ConstantOp(identifier_attr)

            mlir_args = (identifier_op,) + in_args

            if tmpBuf != 0:
                sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
                out_types = tuple(list(out_types) + [sa])

            i32_type = ir.IntegerType.get_signless(32)
            custom_call = stablehlo.CustomCallOp(
                out_types,
                mlir_args,
                call_target_name="jaxzyme.primal",
                backend_config=ir.StringAttr.get("backend"),
                api_version=ir.IntegerAttr.get(i32_type, 3),
            )
            results = tuple(t for t in custom_call.results)

            if tmpBuf != 0:
                results = results[:-1]

            if len(results) != len(out_shapes):
                print(tmpBuf, out_shapes, "\n", results, "\n", str(custom_call))
            assert len(results) == len(out_shapes)

        results2 = []
        residx = 0
        for k in sorted(out_idx_map):
            v = out_idx_map[k]
            if v < 0 or v in kept:
                results2.append(results[residx])
                residx += 1
            else:
                z = make_mlir_zero(orig_types[v])
                results2.append(z)

        results = tuple(results2)
    else:
        assert len(ctx.module_context.platforms) == 1
        identifier, tmpBuf = enzyme_call.create_enzyme_kernel(
            source,
            fn,
            out_shapes,
            in_shapes,
            argv,
            enzyme_call.ABI.Primal,
            lang,
            pipeline_options.xla_runtime(),
            pass_pipeline,
            ctx.module_context.platforms[0],
        )
        identifier_attr = jax_mlir.dense_int_elements([identifier])
        identifier_op = stablehlo.ConstantOp(identifier_attr)

        mlir_args = (identifier_op,) + in_args

        if tmpBuf != 0:
            sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
            out_types = out_types + (sa,)

        i32_type = ir.IntegerType.get_signless(32)
        custom_call = stablehlo.CustomCallOp(
            out_types,
            mlir_args,
            call_target_name="jaxzyme.primal",
            backend_config=ir.StringAttr.get("backend"),
            api_version=ir.IntegerAttr.get(i32_type, 3),
        )

        results = custom_call.results
        results = tuple(t for t in custom_call.results)

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
    pipeline_options,
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(
        itertools.chain(*map(lambda x: _aval_to_ir_types(ctx, x), ctx.avals_out))
    )

    out_shapes = list(map(maketup, out_types[::2]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        in_tree, _, _, mfunc, jit_options = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        avals_in, avals_inkw = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in[::2])
        jit_options = dict(jit_options)
        lowered_func = lower(jax.jit(mfunc, **jit_options), avals_in, kwargs=avals_inkw)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = mhlo.operation.get_asm(enable_debug_info=True)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i // 2 in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    assert len(ctx.module_context.platforms) == 1
    identifier, tmpBuf = enzyme_call.create_enzyme_kernel(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        enzyme_call.ABI.Forward,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
        ctx.module_context.platforms[0],
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    mlir_args = (identifier_op,) + in_args

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        out_types = out_types + (sa, sa)

    i32_type = ir.IntegerType.get_signless(32)
    custom_call = stablehlo.CustomCallOp(
        out_types,
        mlir_args,
        call_target_name="jaxzyme.fwd",
        backend_config=ir.StringAttr.get("backend"),
        api_version=ir.IntegerAttr.get(i32_type, 3),
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
    pipeline_options,
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(
        itertools.chain(*map(lambda x: _aval_to_ir_types(ctx, x), ctx.avals_out))
    )

    out_shapes = list(map(maketup, out_types[: len(out_types) - 1]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        in_tree, _, _, mfunc, jit_options = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        avals_in, avals_inkw = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in)
        jit_options = dict(jit_options)
        lowered_func = lower(jax.jit(mfunc, **jit_options), avals_in, kwargs=avals_inkw)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = mhlo.operation.get_asm(enable_debug_info=True)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    assert len(ctx.module_context.platforms) == 1
    identifier, tmpBuf = enzyme_call.create_enzyme_kernel(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        enzyme_call.ABI.Augmented,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
        ctx.module_context.platforms[0],
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        out_types = out_types + (sa,)

    mlir_args = (identifier_op,) + in_args

    i32_type = ir.IntegerType.get_signless(32)
    custom_call = stablehlo.CustomCallOp(
        out_types,
        mlir_args,
        call_target_name="jaxzyme.aug",
        backend_config=ir.StringAttr.get("backend"),
        api_version=ir.IntegerAttr.get(i32_type, 3),
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
    pipeline_options,
) -> Sequence[ir.Value]:
    del in_shapes

    pre_in_types = tuple(
        itertools.chain(*map(lambda x: _aval_to_ir_types(ctx, x), ctx.avals_out))
    )

    in_shapes = list(map(maketup, pre_in_types))

    out_shapes = list(map(lambda x: maketup(x.type), args_flat[1:]))

    in_args = (*args_flat,)

    rev_return_types = pre_in_types

    kept = None
    if lang == LANG_MHLO:
        in_tree, _, _, mfunc, jit_options = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        avals_in, avals_inkw = jax.tree_util.tree_unflatten(in_tree, ctx.avals_out)
        jit_options = dict(jit_options)
        lowered_func = lower(jax.jit(mfunc, **jit_options), avals_in, kwargs=avals_inkw)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = mhlo.operation.get_asm(enable_debug_info=True)
        kept = lowered_func.compile()._executable._kept_var_idx
        # in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]
        rev_return_types = tuple(
            retty for (i, retty) in enumerate(rev_return_types) if i in kept
        )

    argv = tuple(argv) + ("-resource-dir", resource_dir()) + cflags()
    assert len(ctx.module_context.platforms) == 1
    identifier, tmpBuf = enzyme_call.create_enzyme_kernel(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        enzyme_call.ABI.Reverse,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
        ctx.module_context.platforms[0],
    )
    identifier_attr = jax_mlir.dense_int_elements([identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)

    mlir_args = (identifier_op,) + in_args

    if tmpBuf != 0:
        sa = ir.RankedTensorType.get((tmpBuf,), ir.IntegerType.get_signless(8))
        rev_return_types = rev_return_types + (sa,)

    i32_type = ir.IntegerType.get_signless(32)
    custom_call = stablehlo.CustomCallOp(
        rev_return_types,
        mlir_args,
        call_target_name="jaxzyme.rev",
        backend_config=ir.StringAttr.get("backend"),
        api_version=ir.IntegerAttr.get(i32_type, 3),
    )
    results = custom_call.results
    if tmpBuf != 0:
        results = results[:-1]
    if kept is not None:
        results = []
        cur_idx = 0
        for i, ty in enumerate(pre_in_types):
            if i in kept:
                results.append(custom_call.results[cur_idx])
                cur_idx += 1
            else:
                results.append(make_mlir_zero(ty))
    return results


def ffi_call(
    *args,
    out_shapes: Sequence[jax.core.ShapedArray],
    source,
    fn: str = "f",
    argv: tuple[str] = (),
    lang: int = LANG_CPP,
    pipeline_options=DefaultCPPPipeline,
):
    assert isinstance(source, str) or len(source) == 5
    return _enzyme_primal_p.bind(
        *args,
        source=source,
        fn=fn,
        argv=argv,
        out_shapes=tuple(out_shapes),
        lang=lang,
        pipeline_options=pipeline_options,
    )


def to_jax_type(mlir_type):
    import jax._src.interpreters.mlir

    et = mlir_type.element_type
    for jtype, mcall in jax._src.interpreters.mlir._dtype_to_ir_type.items():
        mtype = mcall()  # mlir_type.context)
        if mtype == et:
            return jax.core.ShapedArray(mlir_type.shape, jtype)
    assert False


def hlo_call(
    *args,
    source: str,
    argv: tuple[str] = (),
    passes: str = "",
):
    fn = "main"
    with jax_mlir.make_ir_context():
        nmod = ir.Module.parse(source)
        func = None
        names = []
        for f in nmod.body:
            names.append(f.sym_name.value)
            if f.sym_name.value == fn:
                func = f
        if func is None:
            raise AssertionError(
                f"Could not find desired function {fn} options are {names}"
            )
        in_tys = list(
            map(lambda x: to_jax_type(x.type), func.regions[0].blocks[0].arguments)
        )
        out_shapes = list(
            map(
                lambda x: to_jax_type(x.type),
                func.regions[0]
                .blocks[0]
                .operations[len(func.regions[0].blocks[0].operations) - 1]
                .operands,
            )
        )
        args_flat, in_tree = jax.tree_util.tree_flatten(args)
        assert len(args_flat) == len(in_tys)
        for jarg, hloty in zip(args_flat, in_tys):
            assert jarg.shape == hloty.shape
            assert jarg.dtype == hloty.dtype

    mfunc = source
    jit_options = {}
    out_idx_map = {i: -1 for (i, v) in enumerate(out_shapes)}
    in_idx_map = {i: i for (i, v) in enumerate(in_tys)}

    return _enzyme_primal_p.bind(
        *args,
        source=(
            in_tree,
            tuple(in_idx_map.items()),
            tuple(out_idx_map.items()),
            mfunc,
            tuple(jit_options.items()),
        ),
        fn=fn,
        argv=argv,
        out_shapes=tuple(out_shapes),
        lang=LANG_MHLO,
        pipeline_options=JaXPipeline(passes),
    )


_JULIA_WORKER_SCRIPT = r"""
using Base64: base64decode, base64encode
using Reactant

const _ORIG_STDOUT = stdout
redirect_stdout(stderr)
Reactant.set_default_backend("cpu")

const _DTYPE_MAP = Dict{String, DataType}(
    "bool" => Bool,
    "int8" => Int8,
    "int16" => Int16,
    "int32" => Int32,
    "int64" => Int64,
    "uint8" => UInt8,
    "uint16" => UInt16,
    "uint32" => UInt32,
    "uint64" => UInt64,
    "float16" => Float16,
    "bfloat16" => Core.BFloat16,
    "float32" => Float32,
    "float64" => Float64,
    "complex64" => ComplexF32,
    "complex128" => ComplexF64,
)

function _make_context()
    if isdefined(Reactant, :ReactantContext)
        return Reactant.ReactantContext(; threading=false)
    else
        ctx = Reactant.MLIR.IR.Context(Reactant.registry[]; threading=false)
        if isdefined(Reactant, :register_enzymexla_dialects)
            Reactant.register_enzymexla_dialects(ctx)
        end
        return ctx
    end
end

function _normalize_result(x::Reactant.TracedRArray{T,N}, seen::IdDict) where {T,N}
    if haskey(seen, x)
        x = Reactant.TracedRArray{T,N}((), x.mlir_data, size(x))
    end
    seen[x] = true
    return x
end

function _normalize_result(x::Reactant.TracedRNumber{T}, seen::IdDict) where {T}
    if haskey(seen, x)
        x = Reactant.TracedRNumber{T}((), x.mlir_data)
    end
    seen[x] = true
    return x
end

function _normalize_result(x::AbstractArray{<:Reactant.TracedRNumber}, seen::IdDict)
    if x isa Array{<:Reactant.TracedRNumber}
        return _normalize_result(Reactant.aos_to_soa(x), seen)
    else
        return _normalize_result(Reactant.materialize_traced_array(x), seen)
    end
end

function _normalize_result(x::AbstractVector, seen::IdDict)
    if !(eltype(x) <: Number)
        return Tuple(_normalize_result(elem, seen) for elem in x)
    end
    T = Reactant.unwrapped_eltype(eltype(x))
    traced = Reactant.TracedUtils.promote_to(Reactant.TracedRArray{T, 1}, x)
    return _normalize_result(traced, seen)
end

function _normalize_result(x::AbstractArray, seen::IdDict)
    T = Reactant.unwrapped_eltype(eltype(x))
    N = ndims(x)
    traced = Reactant.TracedUtils.promote_to(Reactant.TracedRArray{T, N}, x)
    return _normalize_result(traced, seen)
end

function _normalize_result(x::Number, seen::IdDict)
    traced = Reactant.TracedUtils.promote_to(Reactant.TracedRNumber{typeof(x)}, x)
    return _normalize_result(traced, seen)
end

function _normalize_result(x::Tuple, seen::IdDict)
    return map(elem -> _normalize_result(elem, seen), x)
end

function _normalize_result(x::NamedTuple, seen::IdDict)
    return map(elem -> _normalize_result(elem, seen), Tuple(x))
end

_normalize_result(::Nothing, ::IdDict) = ()
_normalize_result(x, ::IdDict) = x

function _normalize_top(res)
    seen = IdDict()
    return _normalize_result(res, seen)
end

function _make_input(dtype_str::AbstractString, dims::Vector{Int}, reverse_dims::Bool)
    T = get(_DTYPE_MAP, String(dtype_str), nothing)
    T === nothing && error("Unsupported JAX dtype for julia_call: $(dtype_str)")
    if isempty(dims)
        return Reactant.ConcreteRNumber{T}(zero(T))
    else
        jl_shape = reverse_dims ? Tuple(reverse(dims)) : Tuple(dims)
        return Reactant.to_rarray(zeros(T, jl_shape))
    end
end

function _compile_in_world(f_obj::F, rargs::Vector{Any}, reverse_dims::Bool) where {F}
    wrapped_fn = function (xs...)
        res = f_obj(xs...)
        return Base.invokelatest(_normalize_top, res)
    end

    ctx = _make_context()
    try
        res_tuple = Reactant.Compiler.compile_mlir(
            ctx,
            wrapped_fn,
            Tuple(rargs);
            compile_options=Reactant.CompileOptions(; donated_args=:none),
            drop_unsupported_attributes=true,
            shardy_passes=:to_mhlo_shardings,
            do_transpose=reverse_dims,
            argprefix=:julia_args,
            args_in_result=:result,
            try_zip_on_failure=false,
        )
        mlir_mod = res_tuple[1]
        mlir_str = string(mlir_mod)
        Reactant.MLIR.IR.dispose(mlir_mod)
        return mlir_str
    finally
        Reactant.MLIR.IR.dispose(ctx)
    end
end

function _handle_request(line::AbstractString)
    parts = split(chomp(line), '\t')
    length(parts) == 4 || error("Invalid request format")
    source = String(base64decode(parts[1]))
    fn_name = String(base64decode(parts[2]))
    reverse_dims = parts[3] == "1"
    args_spec = parts[4]

    mod = Module()
    Core.eval(mod, :(using Reactant))
    if !isempty(strip(source))
        Base.include_string(mod, source, "julia_call.jl")
    end
    f_obj = Core.eval(mod, Meta.parse(fn_name))

    rargs = Any[]
    if !isempty(args_spec)
        for item in split(args_spec, ';')
            isempty(item) && continue
            dt_str, shape_str = split(item, ':')
            dims = isempty(shape_str) ? Int[] : [parse(Int, d) for d in split(shape_str, ',')]
            push!(rargs, _make_input(dt_str, dims, reverse_dims))
        end
    end

    return Base.invokelatest(_compile_in_world, f_obj, rargs, reverse_dims)
end

println(_ORIG_STDOUT, "__ENZYME_JULIA_READY__")
flush(_ORIG_STDOUT)

while !eof(stdin)
    line = readline(stdin)
    isempty(strip(line)) && continue
    try
        mlir = _handle_request(line)
        println(_ORIG_STDOUT, "__ENZYME_JULIA_RESPONSE__:OK\t", base64encode(mlir))
        flush(_ORIG_STDOUT)
    catch e
        err_msg = sprint(showerror, e, catch_backtrace())
        println(_ORIG_STDOUT, "__ENZYME_JULIA_RESPONSE__:ERR\t", base64encode(err_msg))
        flush(_ORIG_STDOUT)
    end
end
"""

_julia_workers = {}
_julia_mlir_cache = {}


def _get_real_home() -> str:
    env_home = os.environ.get("HOME", "")
    if env_home and os.path.isdir(os.path.join(env_home, ".julia")):
        return env_home
    if os.path.isdir("/github/home/.julia"):
        return "/github/home"
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")


def _manifest_julia_channel(project_dir: str, real_home: str) -> str | None:
    manifest_path = os.path.join(project_dir, "Manifest.toml")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(15):
                line = f.readline()
                if not line:
                    break
                if line.startswith("julia_version"):
                    ver = line.split("=", 1)[1].strip().strip('"')
                    parts = ver.split(".")
                    if len(parts) >= 2:
                        maj_min = f"{parts[0]}.{parts[1]}"
                        if os.path.isdir(os.path.join(real_home, ".juliaup")):
                            return f"+{maj_min}"
    except Exception:
        pass
    return None


def _discover_reactant_project(real_home: str) -> tuple[str | None, str | None]:
    # 1. Check explicit environment variables first.
    for env_key in ("ENZYME_JULIA_PROJECT", "JULIA_PROJECT"):
        proj = os.environ.get(env_key)
        if proj and not proj.startswith("@"):
            proj_dir = proj if os.path.isdir(proj) else os.path.dirname(proj)
            return proj, _manifest_julia_channel(proj_dir, real_home)

    # 2. Check if any default ~/.julia/environments/vX.Y/Project.toml has Reactant.
    env_root = os.path.join(real_home, ".julia", "environments")
    if os.path.isdir(env_root):
        for name in sorted(os.listdir(env_root), reverse=True):
            pt = os.path.join(env_root, name, "Project.toml")
            if os.path.isfile(pt):
                try:
                    with open(pt, "r", encoding="utf-8", errors="ignore") as f:
                        if "Reactant" in f.read():
                            chan = (
                                f"+{name[1:]}"
                                if name.startswith("v")
                                and os.path.isdir(os.path.join(real_home, ".juliaup"))
                                else None
                            )
                            return os.path.join(env_root, name), chan
                except Exception:
                    pass

    # 3. Scan ~/.julia/logs/manifest_usage.toml for a precompiled Reactant environment.
    usage_toml = os.path.join(real_home, ".julia", "logs", "manifest_usage.toml")
    candidates = []
    if os.path.isfile(usage_toml):
        try:
            with open(usage_toml, "r", encoding="utf-8", errors="ignore") as f:
                current_manifest = None
                for line in f:
                    line = line.strip()
                    if line.startswith('[["') and line.endswith('"]]'):
                        current_manifest = line[3:-3]
                    elif line.startswith("time =") and current_manifest:
                        ts = line.split("=", 1)[1].strip()
                        candidates.append((ts, current_manifest))
                        current_manifest = None
        except Exception:
            pass

    scored_projects = []
    for ts, manifest_path in candidates:
        if not os.path.isfile(manifest_path):
            continue
        proj_dir = os.path.dirname(manifest_path)
        if os.path.basename(proj_dir) in ("test", "deps", "docs"):
            continue
        proj_toml = os.path.join(proj_dir, "Project.toml")
        if not os.path.isfile(proj_toml):
            continue
        try:
            with open(proj_toml, "r", encoding="utf-8", errors="ignore") as f:
                pt_text = f.read()
            if (
                "3c362404-f566-11ee-1572-e11a4b42c853" not in pt_text
                and "Reactant" not in pt_text
            ):
                continue
            with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
                mf_full = f.read()
            if "Reactant_jll" not in mf_full:
                continue
            if "[[deps.Reactant_jll]]" in mf_full:
                sec = mf_full.split("[[deps.Reactant_jll]]", 1)[1].split("[[", 1)[0]
                if "path =" in sec:
                    dev_path = (
                        sec.split("path =", 1)[1].splitlines()[0].strip().strip('"')
                    )
                    if not os.path.isdir(dev_path):
                        continue
            chan = _manifest_julia_channel(proj_dir, real_home)
            if chan is not None:
                ver_dir = f"v{chan[1:]}"
                compiled_dir = os.path.join(
                    real_home, ".julia", "compiled", ver_dir, "Reactant"
                )
                if os.path.isdir(compiled_dir):
                    src_file = os.path.join(proj_dir, "src", "Reactant.jl")
                    src_entry = src_file.encode("utf-8")
                    src_mtime = (
                        os.path.getmtime(src_file) if os.path.isfile(src_file) else 0.0
                    )
                    best_ji_mtime = -1.0
                    if not os.path.isfile(src_file):
                        for fn in os.listdir(compiled_dir):
                            if fn.endswith((".so", ".dylib", ".dll")):
                                best_ji_mtime = max(
                                    best_ji_mtime,
                                    os.path.getmtime(os.path.join(compiled_dir, fn)),
                                )
                    else:
                        for fn in os.listdir(compiled_dir):
                            if not fn.endswith(".ji"):
                                continue
                            base = fn[:-3]
                            has_lib = any(
                                os.path.isfile(os.path.join(compiled_dir, base + ext))
                                for ext in (".so", ".dylib", ".dll")
                            )
                            if not has_lib:
                                continue
                            ji_path = os.path.join(compiled_dir, fn)
                            ji_mtime = os.path.getmtime(ji_path)
                            if ji_mtime < src_mtime:
                                continue
                            try:
                                with open(ji_path, "rb") as jf:
                                    if src_entry in jf.read(65536):
                                        best_ji_mtime = max(best_ji_mtime, ji_mtime)
                            except Exception:
                                pass
                    if best_ji_mtime > 0:
                        scored_projects.append((best_ji_mtime, ts, proj_dir, chan))
        except Exception:
            continue

    if scored_projects:
        scored_projects.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored_projects[0][2], scored_projects[0][3]

    return None, None


def _resolve_juliaup_binary(real_home: str, channel: str | None) -> str | None:
    import json

    juliaup_dir = os.path.join(real_home, ".julia", "juliaup")
    cfg_path = os.path.join(juliaup_dir, "juliaup.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        ch_name = (
            channel[1:]
            if channel and channel.startswith("+")
            else cfg.get("Default")
        )
        if not ch_name:
            return None
        ver_key = cfg.get("InstalledChannels", {}).get(ch_name, {}).get("Version")
        if not ver_key:
            return None
        rel_bin = cfg.get("InstalledVersions", {}).get(ver_key, {}).get("BinaryPath")
        if not rel_bin:
            return None
        abs_bin = os.path.normpath(os.path.join(juliaup_dir, rel_bin))
        if os.path.isfile(abs_bin):
            return abs_bin
    except Exception:
        pass
    return None


def _find_julia_cmd(
    julia_project: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    import shutil

    real_home = _get_real_home()
    env = os.environ.copy()
    real_depot = os.path.join(real_home, ".julia")
    if os.path.isdir(real_depot):
        if not os.path.isdir(os.path.join(env.get("HOME", ""), ".julia")):
            env["HOME"] = real_home
        if "JULIA_DEPOT_PATH" not in env:
            # Put a writable dir first when running in a read-only sandbox.
            if not os.access(real_depot, os.W_OK):
                writable_depot = os.path.join(
                    env.get("TEST_TMPDIR") or tempfile.gettempdir(),
                    "enzyme_julia_depot",
                )
                os.makedirs(writable_depot, exist_ok=True)
                env["JULIA_DEPOT_PATH"] = f"{writable_depot}:{real_depot}:"
            else:
                env["JULIA_DEPOT_PATH"] = f"{real_depot}:"

    channel = env.get("ENZYME_JULIA_CHANNEL")
    if julia_project is not None:
        proj_dir = (
            julia_project
            if os.path.isdir(julia_project)
            else os.path.dirname(julia_project)
        )
        if not channel:
            channel = _manifest_julia_channel(proj_dir, real_home)
    else:
        julia_project, discovered_channel = _discover_reactant_project(real_home)
        if not channel:
            channel = discovered_channel

    julia_bin = env.get("ENZYME_JULIA_BIN")
    if not julia_bin and env.get("JULIA_BINDIR"):
        cand = os.path.join(env["JULIA_BINDIR"], "julia")
        if os.path.isfile(cand):
            julia_bin = cand
    used_direct_juliaup_bin = False
    if not julia_bin:
        direct_bin = _resolve_juliaup_binary(real_home, channel)
        if direct_bin:
            julia_bin = direct_bin
            used_direct_juliaup_bin = True
    if not julia_bin:
        julia_bin = shutil.which("julia")
    if not julia_bin:
        cand = os.path.join(real_home, ".juliaup", "bin", "julia")
        if os.path.isfile(cand):
            julia_bin = cand
    if not julia_bin:
        raise RuntimeError(
            "Could not find `julia` executable. "
            "Set ENZYME_JULIA_BIN or add julia to PATH."
        )

    cmd = [julia_bin]
    if channel and "juliaup" in julia_bin and not used_direct_juliaup_bin:
        cmd.append(channel)
    cmd.extend(["--startup-file=no", "--history-file=no", "--color=no"])
    if julia_project:
        cmd.append(f"--project={julia_project}")
    return cmd, env


class _JuliaReactantWorker:
    def __init__(self, julia_project: str | None = None):
        import atexit
        import subprocess

        self.cmd, self.env = _find_julia_cmd(julia_project)
        self.proc = subprocess.Popen(
            self.cmd + ["-e", _JULIA_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.env,
        )
        atexit.register(self.close)
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr_out = self.proc.stderr.read() if self.proc.stderr else ""
                cmd_str = " ".join(self.cmd)
                raise RuntimeError(
                    f"Failed to start Julia Reactant worker ({cmd_str}):\n{stderr_out}"
                )
            if line.strip() == "__ENZYME_JULIA_READY__":
                break

    def compile(
        self,
        source: str,
        fn: str,
        in_shapes_dtypes: Sequence[tuple[str, tuple[int, ...]]],
        reverse_dims: bool,
    ) -> str:
        import base64

        src_b64 = base64.b64encode(source.encode("utf-8")).decode("ascii")
        fn_b64 = base64.b64encode(fn.encode("utf-8")).decode("ascii")
        rev_flag = "1" if reverse_dims else "0"
        spec_items = []
        for dt_str, shape in in_shapes_dtypes:
            dims_str = ",".join(str(int(d)) for d in shape)
            spec_items.append(f"{dt_str}:{dims_str}")
        args_spec = ";".join(spec_items)

        req_line = f"{src_b64}\t{fn_b64}\t{rev_flag}\t{args_spec}\n"
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(req_line)
        self.proc.stdin.flush()

        while True:
            resp_line = self.proc.stdout.readline()
            if not resp_line:
                stderr_out = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(
                    f"Julia Reactant worker exited unexpectedly:\n{stderr_out}"
                )
            if resp_line.startswith("__ENZYME_JULIA_RESPONSE__:"):
                payload = resp_line[len("__ENZYME_JULIA_RESPONSE__:") :].rstrip("\r\n")
                status, data_b64 = payload.split("\t", 1)
                text = base64.b64decode(data_b64.encode("ascii")).decode("utf-8")
                if status == "OK":
                    return text
                raise ValueError(
                    f"Failed to compile Julia function `{fn}` with Reactant.jl:\n{text}"
                )

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def _compile_julia_to_mlir(
    source: str | None,
    fn: str | Any,
    in_shapes_dtypes: tuple[tuple[str, tuple[int, ...]], ...],
    reverse_dims: bool = False,
    julia_project: str | None = None,
) -> str:
    if not isinstance(fn, str):
        raise ValueError(
            "`fn` must be a Julia function name string when calling julia_call"
        )

    source_str = source or ""
    if not source_str.strip() and not fn.strip():
        raise ValueError(
            "Either `source` or a valid Julia function name `fn` must be provided"
        )

    cache_key = (source_str, fn, in_shapes_dtypes, bool(reverse_dims), julia_project)
    if cache_key in _julia_mlir_cache:
        return _julia_mlir_cache[cache_key]

    worker = _julia_workers.get(julia_project)
    if worker is None or worker.proc.poll() is not None:
        worker = _JuliaReactantWorker(julia_project=julia_project)
        _julia_workers[julia_project] = worker

    mlir_str = worker.compile(source_str, fn, in_shapes_dtypes, bool(reverse_dims))
    _julia_mlir_cache[cache_key] = mlir_str
    return mlir_str


def julia_call(
    *args,
    source: str | None = None,
    fn: str = "f",
    out_shapes: Sequence[jax.core.ShapedArray] | jax.core.ShapedArray | None = None,
    argv: tuple[str, ...] = (),
    passes: str = "",
    reverse_dims: bool = False,
    julia_project: str | None = None,
):
    """Calls a Julia function from JAX by lowering it to StableHLO MLIR via Reactant.jl.

    ``source`` provides Julia code defining ``fn`` (evaluated in a module with
    ``using Reactant``). If ``source`` is omitted, ``fn`` can name any Julia
    function expression accessible in that module.

    By default (``reverse_dims=False``), a JAX array of shape
    ``(D_0, ..., D_{R-1})`` is traced as a Julia ``TracedRArray`` with matching
    dimensions ``(D_0, ..., D_{R-1})``. Setting ``reverse_dims=True`` reverses
    input and output array dimensions between row-major JAX and column-major
    Julia ``(D_{R-1}, ..., D_0)``.

    Output shapes and dtypes are inferred automatically from the compiled
    Reactant.jl MLIR module; if ``out_shapes`` is provided, it is validated
    against the inferred output types.
    """
    args_flat, _ = jax.tree_util.tree_flatten(args)
    in_shapes_dtypes = tuple(
        (str(jnp.dtype(a.dtype)), tuple(int(d) for d in a.shape)) for a in args_flat
    )
    mlir_source = _compile_julia_to_mlir(
        source=source,
        fn=fn,
        in_shapes_dtypes=in_shapes_dtypes,
        reverse_dims=reverse_dims,
        julia_project=julia_project,
    )

    if out_shapes is not None:
        if isinstance(out_shapes, (jax.core.ShapedArray, jax.ShapeDtypeStruct)):
            expected_out_shapes = (out_shapes,)
        else:
            expected_out_shapes = tuple(out_shapes)

        with jax_mlir.make_ir_context():
            nmod = ir.Module.parse(mlir_source)
            entry_func = None
            for f in nmod.body:
                if hasattr(f, "sym_name") and f.sym_name.value == "main":
                    entry_func = f
                    break
            assert entry_func is not None
            inferred_out_shapes = tuple(
                to_jax_type(op.type)
                for op in entry_func.regions[0]
                .blocks[0]
                .operations[len(entry_func.regions[0].blocks[0].operations) - 1]
                .operands
            )

        if len(expected_out_shapes) != len(inferred_out_shapes):
            raise ValueError(
                "julia_call out_shapes count mismatch: expected "
                f"{len(expected_out_shapes)} outputs, but Reactant.jl inferred "
                f"{len(inferred_out_shapes)} outputs"
            )
        for idx, (exp, inf) in enumerate(zip(expected_out_shapes, inferred_out_shapes)):
            exp_shape = tuple(int(d) for d in exp.shape)
            inf_shape = tuple(int(d) for d in inf.shape)
            exp_dtype = jnp.dtype(exp.dtype)
            inf_dtype = jnp.dtype(inf.dtype)
            if exp_shape != inf_shape or exp_dtype != inf_dtype:
                raise ValueError(
                    f"julia_call out_shapes[{idx}] mismatch: expected "
                    f"ShapedArray({exp_shape}, {exp_dtype}), but Reactant.jl inferred "
                    f"ShapedArray({inf_shape}, {inf_dtype})"
                )

    return hlo_call(
        *args_flat,
        source=mlir_source,
        argv=argv,
        passes=passes,
    )


def cpp_call(
    *args,
    out_shapes: Sequence[jax.core.ShapedArray],
    source: str,
    fn: str = "f",
    argv: tuple[str] = (),
    pipeline_options=DefaultCPPPipeline,
):
    return ffi_call(
        *args,
        source=source,
        fn=fn,
        argv=argv,
        out_shapes=out_shapes,
        lang=LANG_CPP,
        pipeline_options=pipeline_options,
    )


_enzyme_primal_p = Primitive("enzyme_primal")
_enzyme_primal_p.multiple_results = True
_enzyme_primal_p.def_impl(_enzyme_primal_impl)
_enzyme_primal_p.def_abstract_eval(_enzyme_primal_abstract_eval)
jax_mlir.register_lowering(_enzyme_primal_p, _enzyme_primal_lowering)

register_custom_call_target("jaxzyme.primal", enzyme_call.get_callback())

_enzyme_fwd_p = Primitive("enzyme_fwd")
_enzyme_fwd_p.multiple_results = True
_enzyme_fwd_p.def_impl(_enzyme_fwd_impl)
_enzyme_fwd_p.def_abstract_eval(_enzyme_fwd_abstract_eval)
jax_mlir.register_lowering(_enzyme_fwd_p, _enzyme_fwd_lowering)

register_custom_call_target("jaxzyme.fwd", enzyme_call.get_callback())


def enzyme_jvp(arg_primals, arg_tangents, **kwargs):
    # TODO propagate activity info rather than make_zero
    def make_zero(tan, prim):
        return lax.full_like(prim, 0) if type(tan) is ad.Zero else tan

    pipeline_options = kwargs["pipeline_options"]

    shadconv = None
    if pipeline_options.mlir_ad() and kwargs["lang"] == LANG_MHLO:
        in_tree, in_idx_map, out_idx_map, mfunc, jit_options = kwargs["source"]
        in_idx_map = dict(in_idx_map)
        out_idx_map = dict(out_idx_map)
        act_tup = []
        args = []

        avals = {}

        for idx, (v, s) in enumerate(zip(arg_primals, arg_tangents)):
            avals[len(args)] = in_idx_map[idx]
            args.append(v)
            if type(s) is ad.Zero:
                act_tup.append("enzyme_const")
            else:
                act_tup.append("enzyme_dup")
                avals[len(args)] = in_idx_map[idx]
                args.append(s)

        args = tuple(args)
        arg_act_tup = ",".join(act_tup)

        outshapes = kwargs["out_shapes"]
        ret_act_tup = ",".join(["enzyme_dup"] * len(outshapes))
        afterad = (
            "arith-raise{stablehlo=true},enzyme-batch-to-stablehlo, "
            + optimization_passes()
            + ", cse, canonicalize"
        )
        newpasses = (
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + optimization_passes()
            + ", cse,enzyme-wrap{infn=main outfn= retTys="
            + ret_act_tup
            + " argTys="
            + arg_act_tup
            + " mode=ForwardMode},"
            + afterad
        )
        if pipeline_options.pass_pipeline() != "":
            oldpasses = pipeline_options.pass_pipeline()
            if "enzyme-wrap" in oldpasses:
                start = oldpasses.rindex("enzyme-wrap{")
                end = oldpasses.index("}", start)
                prev_passes = oldpasses[:end]
                newpasses = prev_passes + afterad + newpasses + oldpasses[end:]
            else:
                newpasses = newpasses + "," + oldpasses
        pipeline_options = JaXPipeline(newpasses)
        outshapes2 = []
        for o in outshapes:
            outshapes2.append(o)
            outshapes2.append(o)
        out_idx_map2 = {2 * k: v for k, v in out_idx_map.items()} | {
            2 * k + 1: v for k, v in out_idx_map.items()
        }
        source = (
            in_tree,
            tuple(avals.items()),
            tuple(out_idx_map2.items()),
            mfunc,
            jit_options,
        )
        shadconv = ffi_call(
            *args,
            out_shapes=outshapes2,
            source=source,
            fn=kwargs["fn"],
            argv=kwargs["argv"],
            lang=kwargs["lang"],
            pipeline_options=pipeline_options,
        )
    else:
        arg_tangents = tuple(
            make_zero(t, p) for (t, p) in zip(arg_tangents, arg_primals)
        )
        args = tuple(v for t in zip(arg_primals, arg_tangents) for v in t)
        shadconv = _enzyme_fwd_p.bind(
            *args,
            source=kwargs["source"],
            fn=kwargs["fn"],
            argv=kwargs["argv"],
            out_shapes=kwargs["out_shapes"],
            lang=kwargs["lang"],
            pipeline_options=kwargs["pipeline_options"],
        )
    res = (shadconv[0::2], shadconv[1::2])
    return res


ad.primitive_jvps[_enzyme_primal_p] = enzyme_jvp


def jaxify(x):
    return {"float32": 0, "float64": 1}[x.__str__()]


def dejaxify(x):
    return {0: jnp.float32, 1: jnp.float64}[x]


_enzyme_aug_p = Primitive("enzyme_aug")
_enzyme_aug_p.multiple_results = True
_enzyme_aug_p.def_impl(_enzyme_aug_impl)
_enzyme_aug_p.def_abstract_eval(_enzyme_aug_abstract_eval)
jax_mlir.register_lowering(_enzyme_aug_p, _enzyme_aug_lowering)

register_custom_call_target("jaxzyme.aug", enzyme_call.get_callback(), platform="cpu")
register_custom_call_target("jaxzyme.aug", enzyme_call.get_callback(), platform="CUDA")
register_custom_call_target("jaxzyme.aug", enzyme_call.get_callback(), platform="ROCM")
register_custom_call_target("jaxzyme.aug", enzyme_call.get_callback(), platform="tpu")

_enzyme_shadow_aug_p = Primitive("enzyme_shadow_aug")
_enzyme_shadow_aug_p.multiple_results = True
_enzyme_shadow_aug_p.def_impl(_enzyme_shadow_aug_impl)
_enzyme_shadow_aug_p.def_abstract_eval(_enzyme_shadow_aug_abstract_eval)

_enzyme_rev_p = Primitive("enzyme_rev")
_enzyme_rev_p.multiple_results = True
_enzyme_rev_p.def_impl(_enzyme_rev_impl)
_enzyme_rev_p.def_abstract_eval(_enzyme_rev_abstract_eval)
jax_mlir.register_lowering(_enzyme_rev_p, _enzyme_rev_lowering)

register_custom_call_target("jaxzyme.rev", enzyme_call.get_callback(), platform="cpu")
register_custom_call_target("jaxzyme.rev", enzyme_call.get_callback(), platform="CUDA")
register_custom_call_target("jaxzyme.rev", enzyme_call.get_callback(), platform="ROCM")
register_custom_call_target("jaxzyme.rev", enzyme_call.get_callback(), platform="tpu")


def fwd_partial_eval(trace, *args, **kwargs):
    assert len(args) % 2 == 0
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


def primal_partial_eval(trace, *args, **kwargs):
    pipeline_options = kwargs["pipeline_options"]
    if (
        not pipeline_options.mlir_ad()
        or kwargs["lang"] != LANG_MHLO
        or pipeline_options.ad_level() == 0
    ):
        return trace.default_process_primitive(_enzyme_primal_p, args, kwargs)

    _, acts, _ = arg_activity_from_pipeline(pipeline_options.pass_pipeline())

    in_tree, in_idx_map, out_idx_map, mfunc, jit_options = kwargs["source"]
    in_idx_map = dict(in_idx_map)
    out_idx_map = dict(out_idx_map)
    primals = []
    tangents = []
    avals = {}

    for idx, v in enumerate(acts):
        avals[idx] = in_idx_map[len(primals) + len(tangents)]
        primals.append(args[len(primals) + len(tangents)])
        if v == "enzyme_dup":
            tangents.append(args[len(primals) + len(tangents)])

    all_primals_known = all(p.is_known() for p in primals)
    some_tangents_unknown = any(not t.is_known() for t in tangents)

    if not (all_primals_known and some_tangents_unknown):
        return trace.default_process_primitive(_enzyme_primal_p, args, kwargs)

    shadow_aug_args = primals + tangents

    out_shapes = kwargs["out_shapes"]
    out_shapes2 = out_shapes[::2]
    del kwargs["out_shapes"]

    shadows_known = trace.default_process_primitive(
        _enzyme_shadow_aug_p, shadow_aug_args, kwargs | {"out_shapes": out_shapes2}
    )

    passes = pipeline_options.pass_pipeline()
    start = passes.rindex("enzyme-wrap{")
    prev_passes = passes[:start]
    end = passes.index("}", start)
    post_passes = passes[end + 1 :]
    newpasses = prev_passes + post_passes[1:]

    pipeline_options = JaXPipeline(newpasses)

    outmap2 = {k // 2: v for k, v in out_idx_map.items() if k % 2 == 0}
    source = (in_tree, tuple(avals.items()), tuple(outmap2.items()), mfunc, jit_options)

    primalret = trace.default_process_primitive(
        _enzyme_primal_p,
        primals,
        {
            "out_shapes": out_shapes2,
            "source": source,
            "fn": kwargs["fn"],
            "argv": kwargs["argv"],
            "lang": kwargs["lang"],
            "pipeline_options": pipeline_options,
        },
    )
    outs = []
    for p, s in zip(primalret, shadows_known):
        outs.append(p)
        outs.append(s)
    return tuple(outs)


pe.custom_partial_eval_rules[_enzyme_primal_p] = primal_partial_eval


def enzyme_vjp(shadow_rets, *prim_args, **kwargs):
    pipeline_options = kwargs["pipeline_options"]
    if pipeline_options.mlir_ad() and kwargs["lang"] == LANG_MHLO:
        passes = pipeline_options.pass_pipeline()
        start = passes.rindex("enzyme-wrap{")
        prev_passes = passes[:start]
        end = passes.index("}", start)
        post_passes = passes[end + 1 :]
        ad_pass = passes[start : end + 1]

        _, acts, _ = arg_activity_from_pipeline(ad_pass)

        ad_pass = ad_pass.replace("enzyme_dup", "enzyme_active")
        ad_pass = ad_pass.replace("ForwardMode", "ReverseModeCombined")

        shadow_rets2 = tuple(
            sret for (i, sret) in enumerate(shadow_rets) if acts[i] == "enzyme_dup"
        )
        preret, _, postret = ret_activity_from_pipeline(ad_pass)

        shadow_rets2 = []
        ret_act = []
        for i, shad in enumerate(shadow_rets):
            if type(shad) is ad.Zero:
                ret_act.append("enzyme_const")
            else:
                ret_act.append("enzyme_active")
                shadow_rets2.append(shad)
        shadow_rets2 = tuple(shadow_rets2)
        ad_pass = preret + ",".join(ret_act) + postret

        newpasses = (
            prev_passes
            + ad_pass
            + ",arith-raise{stablehlo=true},enzyme-batch-to-stablehlo,canonicalize, remove-unnecessary-enzyme-ops, enzyme-simplify-math, "
            + optimization_passes()
            + ", canonicalize, cse"
            + post_passes
        )

        pipeline_options = JaXPipeline(newpasses)

        in_tree, in_idx_map, out_idx_map, mfunc, jit_options = kwargs["source"]
        in_idx_map = dict(in_idx_map)

        prim_args = prim_args[: len(acts)]

        primal_in_shapes = tuple(
            jax.core.ShapedArray(a.shape, a.dtype) for a in prim_args
        )
        out_shapes2 = [
            shape
            for (i, shape) in enumerate(primal_in_shapes)
            if acts[i] == "enzyme_dup"
        ]

        avals = {}
        outmap = {}
        argidx = 0
        outidx = 0
        for idx, v in enumerate(acts):
            avals[idx] = in_idx_map[argidx]
            if v == "enzyme_dup":
                outmap[outidx] = in_idx_map[argidx]
                outidx += 1
            argidx += 1
            if v == "enzyme_dup":
                argidx += 1

        source = (
            in_tree,
            tuple(avals.items()),
            tuple(outmap.items()),
            mfunc,
            jit_options,
        )

        assert len(outmap) == len(out_shapes2)
        shadconv = _enzyme_primal_p.bind(
            *(prim_args + tuple(shadow_rets2)),
            out_shapes=tuple(out_shapes2),
            source=source,
            fn=kwargs["fn"],
            argv=kwargs["argv"],
            lang=kwargs["lang"],
            pipeline_options=pipeline_options,
        )
        res = tuple(None for _ in prim_args) + tuple(shadconv)
        return res

    del kwargs["out_shapes"]
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


def export(outfile, func, *args, argv=(), jit_options={}):
    def zero_like(arg):
        if arg.dtype == jax.float0:
            return arg
        else:
            return jnp.zeros(arg.shape, dtype=arg.dtype)

    args_flat, in_tree = jax.tree_util.tree_flatten(args)
    in_shapes = [absmaketup(a) for a in args_flat]
    jitres = jax.jit(func, **jit_options)
    out_shape = jitres.eval_shape(*args)
    out_shape_flat, out_tree = jax.tree_util.tree_flatten(out_shape)
    out_shape_flat = [jax.core.ShapedArray(o.shape, o.dtype) for o in out_shape_flat]
    avals_in = jax.tree_util.tree_unflatten(
        in_tree,
        [zero_like(arg) for arg in args_flat],
    )
    lowered_func = lower(jitres, avals_in)
    mhlo = lowered_func.compiler_ir(dialect="stablehlo")
    source = mhlo.operation.get_asm(enable_debug_info=True)
    kept = lowered_func.compile()._executable._kept_var_idx
    in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]
    xla_runtime = False
    pass_pipeline = ""
    lang = LANG_MHLO
    enzyme_call.compile_to_llvm(
        outfile,
        source,
        "",
        list(map(absmaketup, out_shape_flat)),
        in_shapes,
        argv,
        lang,
        xla_runtime,
        pass_pipeline,
    )
    return


def enzyme_jax_ir(
    argv=(), pipeline_options=DefaultJaXPipeline, jit_options={}, inner_jit=True
):
    jit_options2 = {k: v for (k, v) in jit_options.items()}
    if "print_mlir" in jit_options2:
        del jit_options2["print_mlir"]
    jit_options2["inline"] = True

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs):
            args_flat, in_tree = jax.tree_util.tree_flatten((args, kwargs))
            jitres = jax.jit(func, **jit_options2)
            out_shape = jitres.eval_shape(*args, **kwargs)
            in_idxs = {i: i for i in range(len(args_flat))}
            out_shape_flat, out_tree = jax.tree_util.tree_flatten(out_shape)
            out_shape_flat = [
                jax.core.ShapedArray(o.shape, o.dtype) for o in out_shape_flat
            ]
            out_idxs = {i: -1 for i in range(len(out_shape_flat))}

            # Perform jax's dead arg ahead of time dead arg elimination to avoid
            # passing in unnecessary args from our end into xla. Here we emulate
            # compilation first with fake args (to not make a user, and so that
            # this code will get DCE'd / not traced).
            # TODO in the future we should look at mlir to determine what actual values
            # we will need and do dead arg elim ourselves based on ir in advance
            def zero_like(arg):
                if arg.dtype == jax.float0:
                    return arg
                else:
                    return jnp.zeros(arg.shape, dtype=arg.dtype)

            avals_in, avals_kwin = jax.tree_util.tree_unflatten(
                in_tree,
                [zero_like(arg) for arg in args_flat],
            )
            lowered_func = lower(jitres, avals_in, kwargs=avals_kwin)
            kept = lowered_func.compile()._executable._kept_var_idx
            args_flat = [
                arg if i in kept else zero_like(arg)
                for (i, arg) in enumerate(args_flat)
            ]

            out_flat = ffi_call(
                *args_flat,
                source=(
                    in_tree,
                    tuple(in_idxs.items()),
                    tuple(out_idxs.items()),
                    func,
                    tuple(jit_options.items()),
                ),
                fn="",
                out_shapes=out_shape_flat,
                argv=argv,
                lang=LANG_MHLO,
                pipeline_options=pipeline_options,
            )
            return jax.tree_util.tree_unflatten(out_tree, out_flat)

        return jax.jit(wrapped, **jit_options2) if inner_jit else wrapped

    return decorator
