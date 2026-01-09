"""JAX primitives for Enzyme connection."""

from functools import partial
from collections.abc import Callable, Sequence
from typing import Any
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
    enable_scatter_gather_optimization_passes: bool = True,
    enable_pad_optimization_passes: bool = True,
    enable_structured_tensors_passes: bool = False,
    enable_slice_to_batch_passes: bool = False,  # this are somewhat expensive to run
    enable_reduce_slice_fusion_passes: bool = True,
    enable_concat_to_batch_passes: bool = True,
    enable_loop_raising_passes: bool = True,
):
    transform_passes_list = [
        "compare_op_canon<16>",
        "transpose_transpose<16>",
        "broadcast_in_dim_op_canon<16>",
        "convert_op_canon<16>",
        "dynamic_broadcast_in_dim_op_not_actually_dynamic<16>",
        "chained_dynamic_broadcast_in_dim_canonicalization<16>",
        "dynamic_broadcast_in_dim_all_dims_non_expanding<16>",
        "noop_reduce_op_canon<16>",
        "empty_reduce_op_canon<16>",
        "dynamic_reshape_op_canon<16>",
        "get_tuple_element_op_canon<16>",
        "real_op_canon<16>",
        "imag_op_canon<16>",
        "conj_complex_negate<16>",
        "get_dimension_size_op_canon<16>",
        "reshape_op_canon<16>",
        "merge_consecutive_reshapes<16>",
        "transpose_is_reshape<16>",
        "zero_extent_tensor_canon<16>",
        "cse_broadcast_in_dim<16>",
        "cse_slice<16>",
        "cse_transpose<16>",
        "cse_convert<16>",
        "cse_dot_general<16>",
        "cse_reshape<16>",
        "cse_mul<16>",
        "cse_div<16>",
        "cse_add<16>",
        "cse_subtract<16>",
        "cse_min<16>",
        "cse_max<16>",
        "cse_neg<16>",
        "cse_abs<16>",
        "cse_concatenate<16>",
        "cse_compare<16>",
        f"concatenate_op_canon<16>({max_constant_threshold})",
        f"select_op_canon<16>({max_constant_threshold})",
        "add_simplify<16>",
        "sub_simplify<16>",
        "and_simplify<16>",
        "max_simplify<16>",
        "min_simplify<16>",
        "or_simplify<16>",
        "xor_simplify<16>",
        "mul_simplify<16>",
        "div_simplify<16>",
        "rem_simplify<16>",
        "pow_simplify<16>",
        "extend_splat<16>",
        "simplify_extend<16>",
        "simplify_wrap<16>",
        "simplify_rotate<16>",
        "noop_slice<16>",
        "noop_reverse<16>",
        "slice_slice<16>",
        "dynamic_slice_slice<16>",
        "slice_dynamic_slice<16>",
        "shift_right_logical_simplify<16>",
        f"pad_simplify<16>({max_constant_threshold})",
        "slice_simplify<16>",
        "convert_simplify<16>",
        "dynamic_slice_to_static<16>",
        "dynamic_update_slice_elim<16>",
        "concat_to_broadcast<16>",
        "reduce_to_reshape<16>",
        "broadcast_to_reshape<16>",
        "slice_internal",
        f"iota_simplify<16>({max_constant_threshold})",
        f"recognize_from_constant<16>({max_constant_threshold})",
        f"broadcast_in_dim_simplify<16>({max_constant_threshold})",
        "convert_concat<1>",
        "dynamic_update_to_concat<1>",
        "slice_of_dynamic_update<1>",
        "slice_elementwise<1>",
        "dot_reshape_dot<1>",
        "concat_fuse<1>",
        "concat_push_binop_add<1>",
        "concat_push_binop_mul<1>",
        "reduce_concat<1>",
        "slice_concat<1>",
        "concat_slice<1>",
        "select_op_used_within_if<1>",
        "bin_broadcast_splat_add<1>",
        "bin_broadcast_splat_subtract<1>",
        "bin_broadcast_splat_div<1>",
        "bin_broadcast_splat_mul<1>",
        "dot_general_simplify<16>",
        "transpose_simplify<16>",
        "reshape_empty_broadcast<1>",
        "reshape_broadcast<1>",
        "broadcast_reshape<1>",
        "transpose_dot_reorder<1>",
        "dot_transpose<1>",
        "transpose_convolution<1>",
        "convolution_transpose<1>",
        "convert_convert_float<1>",
        "convert_convert_int<1>",
        "reshape_iota<1>",
        "broadcast_reduce<1>",
        "slice_dot_general<1>",
        "if_inline<1>",
        "if_to_select<1>",
        "divide_sqrt_to_multiply_rsqrt<16>",
        "associative_binary_op_reordering<1>",
        "transpose_broadcast_in_dim_to_broadcast_in_dim<16>",
        "replace_neg_add_with_subtract",
        "replace_subtract_neg_with_add",
        "binop_const_simplify",
        "not_select_simplify",
        "common_compare_expression_rewrite",
        "compare_select_simplify",
        "while_simplify<1>(1)",
        "if_remove_unused",
        "transpose_reshape_to_broadcast",
        "reshape_transpose_to_broadcast",
        "dus_dus",
        "dus_dus_concat",
        "abs_positive_simplify",
        "transpose_elementwise_transpose",
        "select_comp_iota_const_simplify<1>",
        "sign_abs_simplify<1>",
        "broadcastindim_is_reshape",
        "reduce_window_wrap<1>",
        "slice_reduce_window<1>",
        "while_deadresult",
        "while_idempotent_dus",
        "while_dus",
        "while_op_induction_replacement",
        "dus_concat",
        "slice_dus_to_concat",
        "hoist_slice",
        "sink_dus",
        "while_induction_reduction",
        "slice_broadcast",
        "associative_common_mul_op_reordering",
        "slice_select_to_select_slice",
        "slice_if",
        "dus_to_i32",
        "slice_extend",
        "concat_wrap",
        "cse_extend<16>",
        "cse_wrap<16>",
        "cse_rotate<16>",
        "cse_rotate<16>",
        "cse_select<16>",
        "concat_concat_axis_swap",
        "concat_concat_to_dus",
        "broadcast_iota_simplify",
        "select_comp_iota_to_dus",
        "compare_cleanup",
        "broadcast_compare",
        "not_compare",
        "broadcast_iota",
        "cse_iota",
        "compare_iota_const_simplify",
        "reshuffle_ands_compares",
        "square_abs_simplify",
        "divide_divide_simplify",
        "concat_reshape_slice",
        "full_reduce_reshape_or_transpose",
        "concat_reshape_reduce",
        "concat_elementwise",
        "reduce_reduce",
        "conj_real",
        "select_broadcast_in_dim",
        "if_op_lift_common_ops",
        "involution_neg_simplify",
        "involution_conj_simplify",
        "involution_not_simplify",
        "real_conj_simplify",
        "conj_complex_simplify",
        "split_convolution_into_reverse_convolution",
        # TODO we want to enable but may cause an infinite compile time
        # "concat_to_onedim_dusslice",
        # "chained_multiply_to_power", # TODO: make it into an optional pass
        "power_multiply_to_power",
        "common_associative_commutative_op_reorder",
        "log_simplify",
        "neg_mul_const_simplify",
        "neg_div_const_simplify",
        "reshape_deletions_broadcast_in_dim_simplify",
        "reshape_insertions_broadcast_in_dim_simplify",
        "dot_general_reshape",
        "widen_wrap",
        "widen_extend",
        "compare_negate_const_simplify",
        "select_simplify",
        "concatenate_broadcast_in_dim",
        "compare_abs",
        # "compare_mul",
        "compare_convert",
        "add_selects",
        # TODO: parameterize based on the device
        "self_subtract_to_convolution_like(0)",
        "self_add_to_convolution_like(0)",
        "self_mul_to_convolution_like(0)",
        "trivial_reduce_window_to_reduce_op",
        "case_to_if",
        "reduce_mul_to_dot_general",
        "split_reduce_add_mul_to_add_dot_general",
        "dot_general_add_distributive_simplify",
        "dot_general_subtract_distributive_simplify",
        "remove_no_ops_from_while_loop",
        "while_is_copy_simplify",
        "split_variadic_scatter_op",
        "dynamic_slice_simplify",
        "enzyme_hlo_unroll(4)",
        "dot_general_only_diagonal_access",
        "divide_negated_operands_simplify",
        "multiply_negated_operands_simplify",
        "factor_scalars_in_dot_general",
        "dot_general_broadcast_in_dim",
        "dot_general_broadcast_in_dim_sort_dims",
        "dus_dynamic_slice_simplify",
        "while_dus_ds_simplify",
        "while_dus_dus_simplify",
        "reshape_slice_reshape",
        "syrk_simplify_output_uplo",
        "dynamic_slice_elementwise",
        "dot_general_remove_batch_dimensions",
        "delete_dims_reduce",
        "reduce_delete_dims",
        "dot_general_insert_dim_contraction_simplification",
        "fuse_reshape_collapse_or_expand_dims_into_reduce",
    ]

    # constant propagation patterns
    transform_passes_list += [
        # unary constant propagation
        "chlo_inf_const_prop<16>",
        "gamma_const_prop<16>",
        "abs_const_prop<16>",
        "log_const_prop<1>",
        "log_plus_one_const_prop<1>",
        "is_finite_const_prop",
        "not_const_prop",
        "neg_const_prop",
        "sqrt_const_prop",
        "rsqrt_const_prop",
        "cos_const_prop",
        "sin_const_prop",
        "exp_const_prop",
        "expm1_const_prop",
        "tanh_const_prop",
        "logistic_const_prop",
        "conj_const_prop",
        "ceil_const_prop",
        "cbrt_const_prop",
        "real_const_prop",
        "imag_const_prop",
        "round_const_prop",
        "round_nearest_even_const_prop",
        "sign_const_prop",
        "floor_const_prop",
        "tan_const_prop",
        # binary constant propagation
        "add_const_prop",
        "and_const_prop",
        "atan2_const_prop",
        "complex_const_prop",
        "div_const_prop",
        "max_const_prop",
        "min_const_prop",
        "mul_const_prop",
        "or_const_prop",
        "pow_const_prop",
        "rem_const_prop",
        "sub_const_prop",
        "xor_const_prop",
        # other constant propagations
        "const_prop_through_barrier<16>",
        f"concat_const_prop<1>({max_constant_threshold})",
        f"dynamic_update_slice_const_prop({max_constant_threshold})",
        "clamp_const_prop",
    ]

    if (
        enable_structured_tensors_passes
    ):  # currently we dont register custom_calls on jax end
        transform_passes_list += ["dot_general_to_syrk"]

    if enable_slice_to_batch_passes:
        transform_passes_list += [
            "dot_general_slice_to_batch",
            "gather_slice_to_batch",
            "iota_slice_to_batch",
            "reduce_slice_to_batch",
            "sort_slice_to_batch",
            "transpose_slice_to_batch",
            "broadcastindim_slice_to_batch",
            "reducewindow_slice_to_batch",
            "elementwise_slice_to_batch",
            "convolution_slice_to_batch",
        ]

    if enable_concat_to_batch_passes:
        transform_passes_list += [
            "concat_insert_dim_dot_general",
            "concat_insert_dim_gather",
            "concat_insert_dim_iota",
            "concat_insert_dim_reduce",
            "concat_insert_dim_sort",
            "concat_insert_dim_reduce_window",
            "concat_insert_dim_elementwise",
            "concat_insert_dim_convolution",
        ]

    if enable_reduce_slice_fusion_passes:
        transform_passes_list += [
            "add_reduce_slice_fusion",
            "mul_reduce_slice_fusion",
            "min_reduce_slice_fusion",
            "max_reduce_slice_fusion",
            "and_reduce_slice_fusion",
            "xor_reduce_slice_fusion",
            "or_reduce_slice_fusion",
        ]

    if enable_loop_raising_passes:
        transform_passes_list += [
            "greedy_while_loop_batch_fission",
            "while_elementwise_reduction_to_reduce",
            "remove_loop_carried_dependencies_from_while_load_operations",
        ]

    if enable_licm_optimization_passes:
        transform_passes_list += [
            "dus_licm(0)",
            "slice_licm(0)",
            "dynamic_slice_licm(0)",
            "elementwise_licm(0)",
            "concatenate_licm(0)",
            "while_licm<1>(1)",
            "transpose_licm(0)",
            "broadcastindim_licm(0)",
            "reshape_licm(0)",
            "dot_general_licm(0)",
            "reduce_licm(0)",
            "reduce_window_licm(0)",
            "reverse_licm(0)",
            "convolution_licm(0)",
            "scatter_licm(0)",
            "gather_licm(0)",
            "iota_licm(0)",
            "extend_licm(0)",
            "wrap_licm(0)",
            "rotate_licm(0)",
        ]

    if enable_scatter_gather_optimization_passes:
        transform_passes_list += [
            "scatter_to_dynamic_update_slice<1>",
            "scatter_multiply_simplify",
            "scatter_div_simplify",
            "scatter_sub_simplify",
            "scatter_add_simplify",
            "unary_elementwise_scatter_simplify",
            "scatter_indices_are_unique",
            "diagonal_tensor_dot_general_rewrite",
            ## const prop patterns
            "scatter_update_computation_const_prop",
            # gather patterns
            "dynamic_gather_op_is_not_dynamic<16>",
            "gather_op_canon<16>",
            "gather_elementwise",
            ## const prop patterns
            "gather_const_prop",
            f"scatter_const_fold({max_constant_threshold})",
            "cse_gather",
            "cse_scatter",
            "gather_of_scatter_simplify",
        ]

    if enable_pad_optimization_passes:
        transform_passes_list += [
            "extend_pad",
            "dus_pad",
            "cse_pad<16>",
            f"pad_simplify<16>({max_constant_threshold})",
            "select_pad_to_dus<1>",
            "and_pad_pad<1>",
            "negative_pad_to_slice<16>",
            "slice_pad<1>",
            "pad_reshape_pad<1>",
            "pad_pad<1>",
            "add_pad_pad_to_concat<1>",
            "concat_pad<1>",
            "reduce_pad<1>",
            "broadcast_pad<1>",
            "zero_product_reshape_pad<1>",
            "mul_zero_pad<1>",
            "div_zero_pad<1>",
            "binop_const_reshape_pad<1>",
            "binop_const_pad_add<1>",
            "binop_const_pad_subtract<1>",
            "binop_const_pad_mul<1>",
            "binop_const_pad_div<1>",
            "binop_binop_pad_pad_add<1>",
            "binop_binop_pad_pad_mul<1>",
            "binop_pad_pad_add<1>",
            "binop_pad_pad_subtract<1>",
            "binop_pad_pad_mul<1>",
            "binop_pad_pad_div<1>",
            "binop_pad_pad_min<1>",
            "binop_pad_pad_max<1>",
            "unary_pad_push_convert<1>",
            "unary_pad_push_tanh<1>",
            "unary_pad_push_exp<1>",
            "concat_to_pad<1>",
            "while_pad_induction_reduction",
            "pad_concat_to_concat_pad",
            "rotate_pad",
            "concat_multipad",
            "speculate_if_pad_to_select",
            "dus_to_dynamic_pad",
            "dynamic_pad_to_pad",
        ]

    if reshape_propagate == "up":
        transform_passes_list += [
            "reshape_concat",
            "reshape_dus",
            "dot_reshape_pad<1>",
            "pad_dot_general<1>(0)",
            # XXX: see https://github.com/EnzymeAD/Enzyme-JAX/issues/1445
            # "pad_dot_general<1>(1)",
            "reshape_pad",
            "reshape_wrap",
            "reshape_rotate",
            "reshape_extend",
            "reshape_slice(1)",
            "reshape_elementwise(1)",
            "reshape_dynamic_slice(1)",
            "delete_dims_broadcast",
        ]
    elif reshape_propagate == "down":
        transform_passes_list += [
            "concat_appending_reshape",
            "slice_reshape",
            "slice_reshape_slice<1>",
            "dynamic_slice_reshape_dynamic_slice",
            "dynamic_slice_reshape_slice",
            "slice_reshape_dynamic_slice",
            "slice_reshape_concat<1>",
            "slice_reshape_elementwise<1>",
            "slice_reshape_dot_general<1>",
            "slice_reshape_pad<1>",
            "elementwise_reshape_like",
            "reshape_elementwise_only_fusible(1)",
        ]
    else:
        raise ValueError("Invalid value for reshape_propagate")

    if transpose_propagate == "up":
        transform_passes_list += [
            "transpose_select",
            "transpose_while",
            "transpose_slice",
            "transpose_concat",
            "transpose_iota",
            "transpose_reduce",
            "transpose_reduce_window",
            "transpose_dus",
            "transpose_pad<1>",
            "transpose_einsum<1>",
            "transpose_wrap",
            "transpose_extend",
            "transpose_rotate",
            "transpose_dynamic_slice",
            "transpose_reverse",
            "transpose_batch_norm_training",
            "transpose_batch_norm_inference",
            "transpose_batch_norm_grad",
            "transpose_if",
            "transpose_fft",
            "transpose_reshape",
        ]
    elif transpose_propagate == "down":
        transform_passes_list += [
            "reorder_elementwise_and_shape_op<16>",
            "elementwise_all_transpose_operands_simplify",
            "dynamic_slice_transpose",
            "slice_transpose",
            "einsum_transpose<1>",
            "slice_reshape_transpose<1>",
            "reduce_transpose_simplify",
            "reverse_transpose",
            "transpose_all_users_slice",
        ]
    else:
        raise ValueError("Invalid value for transpose_propagate")

    if no_nan:
        transform_passes_list += [
            "no_nan_compare_simplify(1)",
            "no_nan_self_sub_simplify(1)",
            "no_nan_add_sub_simplify(1)",
            "no_nan_mul_simplify(1)",
            "no_nan_div_simplify(1)",
            # "no_nan_zero_base_pow_simplify(1)",
        ]
    else:
        transform_passes_list += [
            "no_nan_compare_simplify(0)",
            "no_nan_self_sub_simplify(0)",
            "no_nan_add_sub_simplify(0)",
            "no_nan_mul_simplify(0)",
            "no_nan_div_simplify(0)",
            # "no_nan_zero_base_pow_simplify(0)",
        ]

    if all_finite:
        transform_passes_list += [
            "all_finite_is_finite",
            "all_finite_is_inf",
            "all_finite_is_pos_inf",
            "all_finite_is_neg_inf",
        ]

    transform_passes = ",".join(
        [
            "enzyme-hlo-generate-td{patterns=" + ";".join(transform_passes_list) + "}",
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


# TODO: implement options similar to ones in Reactant for benchmarking
#       currently we mimic the `:all` option from Reactant
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
        (in_tree, _, _, mfunc, jit_options) = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        (avals_in, avals_inkw) = jax.tree_util.tree_unflatten(in_tree, args_flat)
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

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types))
    in_shapes = list(map(lambda x: maketup(x.type), args_flat))
    in_types = list(map(lambda x: x.type, args_flat))

    in_args = (*args_flat,)

    pass_pipeline = pipeline_options.pass_pipeline()

    out_idx_map = {i: -1 for i in range(len(out_shapes))}

    argv = argv + ("-resource-dir", resource_dir()) + cflags()

    if lang == LANG_MHLO:
        (in_tree, in_idx_map, out_idx_map, mfunc, jit_options) = source
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
            (avals_in, avals_inkw) = jax.tree_util.tree_unflatten(in_tree, avals)
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
            ins = ir.InsertionPoint.current
            mod = ins.block.region.owner.parent
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

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types[::2]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat[::2]))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, _, _, mfunc, jit_options) = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        (avals_in, avals_inkw) = jax.tree_util.tree_unflatten(
            in_tree, ctx.avals_in[::2]
        )
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

    out_types = tuple(itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out)))

    out_shapes = list(map(maketup, out_types[: len(out_types) - 1]))

    in_shapes = list(map(lambda x: maketup(x.type), args_flat))

    in_args = (*args_flat,)

    if lang == LANG_MHLO:
        (in_tree, _, _, mfunc, jit_options) = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        (avals_in, avals_inkw) = jax.tree_util.tree_unflatten(in_tree, ctx.avals_in)
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
        itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
    )

    in_shapes = list(map(maketup, pre_in_types))

    out_shapes = list(map(lambda x: maketup(x.type), args_flat[1:]))

    in_args = (*args_flat,)

    rev_return_types = pre_in_types

    kept = None
    if lang == LANG_MHLO:
        (in_tree, _, _, mfunc, jit_options) = source
        if "print_mlir" in jit_options:
            del jit_options["print_mlir"]
        (avals_in, avals_inkw) = jax.tree_util.tree_unflatten(in_tree, ctx.avals_out)
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
        return lax.zeros_like_array(prim) if type(tan) is ad.Zero else tan

    pipeline_options = kwargs["pipeline_options"]

    shadconv = None
    if pipeline_options.mlir_ad() and kwargs["lang"] == LANG_MHLO:
        (in_tree, in_idx_map, out_idx_map, mfunc, jit_options) = kwargs["source"]
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

    (in_tree, in_idx_map, out_idx_map, mfunc, jit_options) = kwargs["source"]
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

        (in_tree, in_idx_map, out_idx_map, mfunc, jit_options) = kwargs["source"]
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

            (avals_in, avals_kwin) = jax.tree_util.tree_unflatten(
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
