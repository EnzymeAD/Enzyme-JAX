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
from jaxlib.mlir.dialects import stablehlo, func
from jax.lib import xla_client

import jax.numpy as jnp

from . import enzyme_call

LANG_CPP = enzyme_call.Language.CPP
LANG_LLVM = enzyme_call.Language.LLVM
LANG_MHLO = enzyme_call.Language.MHLO

from enum import Enum


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


class OldXLAPipeline:
    def xla_runtime(self):
        return False

    def pass_pipeline(self):
        return ""

    def mlir_ad(self):
        return False

    def stablehlo_inject(self):
        return False


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


class NewXLAPipeline:
    def __init__(self, passes=None, mlirad=False):
        if passes is None:
            passes = """
          stablehlo-legalize-to-hlo,
          inline{default-pipeline=canonicalize max-iterations=4},
          expand-hlo-tuples{entry-function=main},
          func.func(mhlo-flatten-tuple),
          xla-legalize-abi,
          func.func(mhlo-test-lower-general-dot),
          func.func(mhlo-broadcast-propagation),
          cse,
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          func.func(xla-sparse-custom-call-to-pack),
          func.func(legalize-sparse-ops{legalize-to-custom-calls=false}),
          func.func(chlo-legalize-to-hlo{
              expand-compositions=true legalize-broadcasts=true}),
          func.func(mhlo-sparse-rewriting),
          func.func(mhlo-legalize-control-flow),
          func.func(mhlo-legalize-dot-general-to-dot),
          hlo-legalize-to-arithmetic,
          func.func(xla-legalize-library-ops),
          func.func(mhlo-expand-ops-simplifier),
          func.func(hlo-canonicalize-scatter),
          func.func(hlo-canonicalize-dot),
          func.func(group-reduction-dimensions{prefer-columns-reductions=true}),
          func.func(hlo-legalize-to-linalg{enable-primitive-ops=false}),
          func.func(lower-index-cast),
          convert-to-signless,
          func.func(shape-simplification),
          func.func(shape-to-shape-lowering),
          convert-shape-to-std,
          func.func(convert-shape-constraints),
          cse,
          resolve-shaped-type-result-dims,
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          func.func(linalg-fuse-elementwise-ops),
          reconcile-unrealized-casts,
          convert-tensor-to-linalg,
          func.func(detensorize-scf-ops),
          func.func(linalg-detensorize{aggressive-mode=true}),
          eliminate-empty-tensors,
          func.func(empty-tensor-to-alloc-tensor),
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          func.func(linalg-generalize-named-ops),
          eliminate-empty-tensors,
          sparsification-and-bufferization,
          sparse-storage-specifier-to-llvm,
          func.func(canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true}),
          func.func(finalizing-bufferize),
          func.func(xla-rewrite-realloc-to-alloc),
          func.func(vectorize-copy),
          func.func(naive-copy-removal),
          func.func(convert-linalg-to-loops),
          cse,
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          buffer-results-to-out-params,
          func.func(promote-buffers-to-stack{
              max-alloc-size-in-bytes=1024
              max-rank-of-allocated-memref=1}),
          func.func(buffer-deallocation),
          convert-bufferization-to-memref,
          func.func(xla-remove-copies-to-out-params),
          cse,
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          func.func(convert-complex-to-standard),
          cse,
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          func.func(convert-vector-to-scf{
              full-unroll=false
              lower-tensors=false
              target-rank=1}),
          func.func(xla-legalize-i1-vector-transfers),
          func.func(xla-convert-memref-element-cast-to-llvm),
          async-func-to-async-runtime,
          xla-rt-export-functions,
          xla-cpu-to-cpu-runtime,
          xla-rt-convert-custom-calls,
          xla-rt-convert-asserts,
          inline{default-pipeline=canonicalize max-iterations=4},
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          cse,
          func.func(xla-math-approximation{oplist=all}),
          func.func(convert-linalg-to-parallel-loops),
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          async-to-async-runtime,
          xla-rt-move-allocas-to-entry-block,
          async-runtime-policy-based-ref-counting,
          func.func(arith-expand{include-bf16=false}),
          func.func(memref-expand),
          func.func(expand-strided-metadata),
          lower-affine,
          func.func(xla-memref-aligned-allocations{alignment=0}),
          xla-rt-to-llvm,
          convert-async-to-llvm,
          generic-host-to-llvm{enable-avx2=false},
          reconcile-unrealized-casts,
          canonicalize{
              max-iterations=10
              max-num-rewrites=-1
              region-simplify=true
              test-convergence=false
              top-down=true},
          cse"""
        assert len(passes) != 0
        self.passes = passes
        self.mlirad = mlirad

    def xla_runtime(self):
        return True

    def pass_pipeline(self):
        return self.passes

    def mlir_ad(self):
        return self.mlirad

    def stablehlo_inject(self):
        return False

    def ad_level(self):
        return self.passes.count("enzyme-wrap")


DefaultPipeline = OldXLAPipeline()  # NewXLAPipeline(None, True)


def pass_pipeline(options):
    if type(options) == type(""):
        return options
    else:
        return


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
    pipeline_options
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
    pipeline_options
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
    pipeline_options
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
    pipeline_options
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
    pipeline_options
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
    pipeline_options
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
    pipeline_options
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
    pipeline_options
) -> Sequence[jax.core.ShapedArray]:
    in_shapes = args_flat

    prev_out_shapes = out_shapes

    out_shapes = [absmaketup(a) for a in out_shapes]

    in_shapes = [absmaketup(a) for a in in_shapes]

    if lang == LANG_MHLO:
        (in_tree, _, _, mfunc) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, args_flat)
        lowered_func = jax.jit(mfunc).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = str(mhlo)
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
    pipeline_options
) -> Sequence[jax.core.ShapedArray]:
    return out_shapes


def _enzyme_rev_abstract_eval(
    *args_flat: jax.core.ShapedArray,
    source,
    fn: str,
    argv: Sequence[str],
    in_shapes,
    lang: enzyme_call.Language,
    pipeline_options
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
    pipeline_options
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types))
    in_shapes = list(map(lambda x: maketup(x.type), args_flat))
    in_types = list(map(lambda x: x.type, args_flat))

    in_args = (*args_flat,)

    pass_pipeline = pipeline_options.pass_pipeline()

    out_idx_map = {i: -1 for i in range(len(out_shapes))}

    argv = argv + ("-resource-dir", resource_dir()) + cflags()

    if lang == LANG_MHLO:
        (in_tree, in_idx_map, out_idx_map, mfunc) = source
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
        avals_in = jax.tree_util.tree_unflatten(in_tree, avals)
        lowered_func = jax.jit(mfunc).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(
            arg
            for (i, arg) in enumerate(in_args)
            if i not in in_idx_map or in_idx_map[i] in kept
        )
        if len(kept) != len(orig_shapes):
            post = ",".join(["enzyme_dup"] * len(kept))
            prev = ",".join(["enzyme_dup"] * len(orig_shapes))
            pass_pipeline = pass_pipeline.replace(prev, post)
            post = ",".join(["enzyme_out"] * len(kept))
            prev = ",".join(["enzyme_out"] * len(orig_shapes))
            pass_pipeline = pass_pipeline.replace(prev, post)

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
            ins = ir.InsertionPoint.current
            mod = ins.block.region.owner.parent
            fns = []
            for f in mod.regions[0].blocks[0]:
                fns.append(f.sym_name.value)

            name, nmod = enzyme_call.run_pass_pipeline(fns, source, pass_pipeline)
            nmod = ir.Module.parse(nmod)
            fn = None
            for f in nmod.body:
                mod.regions[0].blocks[0].append(f)
                if f.sym_name.value == name:
                    fn = f
            results = func.CallOp(fn, list(in_args)).results
            if len(results) != len(out_shapes):
                print(out_shapes, "\n", results, "\n", nmod)
            assert len(results) == len(out_shapes)
        else:
            identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
                source,
                fn,
                out_shapes,
                in_shapes,
                argv,
                enzyme_call.ABI.Primal,
                lang,
                pipeline_options.xla_runtime(),
                pass_pipeline,
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
            results = tuple(t for t in custom_call.results)

            if tmpBuf != 0:
                results = results[:-1]

            if len(results) != len(out_shapes):
                print(tmpBuf, out_shapes, "\n", results, "\n", str(custom_call))
            assert len(results) == len(out_shapes)

        def zero(ty):
            from jax._src.interpreters import mlir

            return mlir.ir_constant(jnp.zeros(ty.shape, dtype=to_jax(ty.element_type)))

        results2 = []
        residx = 0
        for k in sorted(out_idx_map):
            v = out_idx_map[k]
            if v < 0 or v in kept:
                results2.append(results[residx])
                residx += 1
            else:
                z = zero(orig_types[v])
                results2.append(z)
        results = tuple(results2)
    else:
        identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
            source,
            fn,
            out_shapes,
            in_shapes,
            argv,
            enzyme_call.ABI.Primal,
            lang,
            pipeline_options.xla_runtime(),
            pass_pipeline,
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
    pipeline_options
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types[::2]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, _, _, mfunc) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in[::2])
        lowered_func = jax.jit(mfunc).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i // 2 in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        enzyme_call.ABI.Forward,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
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
    pipeline_options
) -> Sequence[ir.Value]:
    del out_shapes

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types[: len(out_types) - 1]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, _, _, mfunc) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in)
        lowered_func = jax.jit(mfunc).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]

    argv = argv + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        enzyme_call.ABI.Augmented,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
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
    pipeline_options
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
        (in_tree, _, _, mfunc) = source
        avals_in = jax.tree_util.tree_unflatten(in_tree, ctx.avals_out)
        lowered_func = jax.jit(mfunc).lower(*avals_in)
        mhlo = lowered_func.compiler_ir(dialect="stablehlo")
        source = str(mhlo)
        kept = lowered_func.compile()._executable._kept_var_idx
        # in_args = tuple(arg for (i, arg) in enumerate(in_args) if i in kept)
        in_shapes = [shape for (i, shape) in enumerate(in_shapes) if i in kept]
        rev_return_types = tuple(
            retty for (i, retty) in enumerate(rev_return_types) if i in kept
        )

    argv = tuple(argv) + ("-resource-dir", resource_dir()) + cflags()
    identifier, tmpBuf = enzyme_call.create_enzyme_cpu_kernel(
        source,
        fn,
        out_shapes,
        in_shapes,
        argv,
        enzyme_call.ABI.Reverse,
        lang,
        pipeline_options.xla_runtime(),
        pipeline_options.pass_pipeline(),
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
    pipeline_options=DefaultPipeline
):
    return _enzyme_primal_p.bind(
        *args,
        source=source,
        fn=fn,
        argv=argv,
        out_shapes=out_shapes,
        lang=lang,
        pipeline_options=pipeline_options
    )


def cpp_call(
    *args,
    out_shapes: Sequence[jax.core.ShapedArray],
    source: str,
    fn: str = "f",
    argv: tuple[str] = (),
    pipeline_options=DefaultPipeline
):
    return ffi_call(
        *args,
        source=source,
        fn=fn,
        argv=argv,
        out_shapes=out_shapes,
        lang=LANG_CPP,
        pipeline_options=pipeline_options
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

    pipeline_options = kwargs["pipeline_options"]

    shadconv = None
    if pipeline_options.mlir_ad() and kwargs["lang"] == LANG_MHLO:
        act_tup = ",".join(["enzyme_dup" for a in arg_primals])
        afterad = "arith-raise{stablehlo=true}, enzyme-hlo-opt, cse, canonicalize"
        newpasses = (
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "enzyme-hlo-opt,cse,enzyme-wrap{infn=main outfn= retTy=enzyme_dup argTys="
            + act_tup
            + " mode=ForwardMode},"
            + afterad
        )
        if pipeline_options.pass_pipeline() != "":
            oldpasses = pipeline_options.pass_pipeline()
            if "enzyme-wrap" in oldpasses:
                start = passes.rindex("enzyme-wrap{")
                end = passes.index("}", start)
                prev_passes = passes[:end]
                newpasses = prev_passes + afterad + newpasses + passes[end:]
            else:
                newpasses = newpasses + "," + oldpasses
        if pipeline_options.stablehlo_inject():
            pipeline_options = JaXPipeline(newpasses)
        else:
            pipeline_options = NewXLAPipeline(newpasses, pipeline_options.mlir_ad())
        outshapes2 = []
        for o in kwargs["out_shapes"]:
            outshapes2.append(o)
            outshapes2.append(o)
        (in_tree, in_idx_map, out_idx_map, mfunc) = kwargs["source"]
        avals = {2 * k: v for k, v in in_idx_map.items()} | {
            2 * k + 1: v for k, v in in_idx_map.items()
        }
        out_idx_map2 = {2 * k: v for k, v in out_idx_map.items()} | {
            2 * k + 1: v for k, v in out_idx_map.items()
        }
        source = (in_tree, avals, out_idx_map2, mfunc)
        shadconv = ffi_call(
            *args,
            out_shapes=outshapes2,
            source=source,
            fn=kwargs["fn"],
            argv=kwargs["argv"],
            lang=kwargs["lang"],
            pipeline_options=pipeline_options
        )
    else:
        shadconv = _enzyme_fwd_p.bind(
            *args,
            source=kwargs["source"],
            fn=kwargs["fn"],
            argv=kwargs["argv"],
            out_shapes=kwargs["out_shapes"],
            lang=kwargs["lang"],
            pipeline_options=kwargs["pipeline_options"]
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


def primal_partial_eval(trace, *args, **kwargs):
    pipeline_options = kwargs["pipeline_options"]
    if (
        not pipeline_options.mlir_ad()
        or kwargs["lang"] != LANG_MHLO
        or pipeline_options.ad_level() == 0
    ):
        return trace.default_process_primitive(_enzyme_primal_p, args, kwargs)

    assert len(args) % 2 == 0
    nr_primals = len(args) // 2
    primals, tangents = args[0::2], args[1::2]
    all_primals_known = all(p.is_known() for p in primals)
    some_tangents_unknown = any(not t.is_known() for t in tangents)

    if not (all_primals_known and some_tangents_unknown):
        return trace.default_process_primitive(_enzyme_primal_p, args, kwargs)

    shadow_aug_args = primals + tangents

    out_shapes = kwargs["out_shapes"]
    out_shapes2 = out_shapes[: len(out_shapes) // 2]
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

    if pipeline_options.stablehlo_inject():
        pipeline_options = JaXPipeline(newpasses)
    else:
        pipeline_options = NewXLAPipeline(newpasses, pipeline_options.mlir_ad())

    (in_tree, in_idx_map, out_idx_map, mfunc) = kwargs["source"]

    avals = {k // 2: v for k, v in in_idx_map.items() if k % 2 == 0}
    outmap2 = {k // 2: v for k, v in out_idx_map.items() if k % 2 == 0}
    source = (in_tree, avals, outmap2, mfunc)

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
    return primalret + shadows_known


pe.custom_partial_eval_rules[_enzyme_primal_p] = primal_partial_eval


def enzyme_vjp(shadow_rets, *prim_args, **kwargs):
    pipeline_options = kwargs["pipeline_options"]
    if pipeline_options.mlir_ad() and kwargs["lang"] == LANG_MHLO:
        prim_args = prim_args[0 : len(prim_args) // 2]

        passes = pipeline_options.pass_pipeline()
        start = passes.rindex("enzyme-wrap{")
        prev_passes = passes[:start]
        end = passes.index("}", start)
        post_passes = passes[end + 1 :]
        ad_pass = passes[start : end + 1]
        ad_pass = ad_pass.replace("enzyme_dup", "enzyme_out")
        ad_pass = ad_pass.replace("ForwardMode", "ReverseModeCombined")
        newpasses = (
            prev_passes
            + ad_pass
            + ",canonicalize, remove-unnecessary-enzyme-ops, enzyme-simplify-math, enzyme-hlo-opt, canonicalize, cse"
            + post_passes
        )

        if pipeline_options.stablehlo_inject():
            pipeline_options = JaXPipeline(newpasses)
        else:
            pipeline_options = NewXLAPipeline(newpasses, pipeline_options.mlir_ad())

        (in_tree, in_idx_map, out_idx_map, mfunc) = kwargs["source"]

        avals = {k // 2: v for k, v in in_idx_map.items() if k % 2 == 0}
        outmap = avals

        primal_in_shapes = tuple((a.shape, jaxify(a.dtype)) for a in prim_args)
        primal_in_shapes = tuple(
            jax.core.ShapedArray(a.shape, a.dtype) for a in prim_args
        )
        out_shapes2 = primal_in_shapes  # out_shapes[:len(out_shapes)/2] +
        source = (in_tree, avals, outmap, mfunc)
        shadconv = _enzyme_primal_p.bind(
            *(prim_args + tuple(shadow_rets)),
            out_shapes=out_shapes2,
            source=source,
            fn=kwargs["fn"],
            argv=kwargs["argv"],
            lang=kwargs["lang"],
            pipeline_options=pipeline_options
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


def enzyme_jax_ir(argv=(), pipeline_options=DefaultPipeline):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @jax.jit
        def wrapped(*args: Any):
            args_flat, in_tree = jax.tree_util.tree_flatten(args)
            out_shape = jax.eval_shape(func, *args)
            in_idxs = {i: i for i in range(len(args_flat))}
            out_shape_flat, out_tree = jax.tree_util.tree_flatten(out_shape)
            out_shape_flat = [
                jax.core.ShapedArray(o.shape, o.dtype) for o in out_shape_flat
            ]
            out_idxs = {i: -1 for i in range(len(out_shape_flat))}
            out_flat = ffi_call(
                *args_flat,
                source=(in_tree, in_idxs, out_idxs, func),
                fn="",
                out_shapes=out_shape_flat,
                argv=argv,
                lang=LANG_MHLO,
                pipeline_options=pipeline_options
            )
            return jax.tree_util.tree_unflatten(out_tree, out_flat)

        return wrapped

    return decorator
