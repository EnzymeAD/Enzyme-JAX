#include "EnzymeXLAOptPasses.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Helper: format a pass with a parenthesized int64 argument.
static std::string passWithArg(const char *name, int64_t arg) {
  std::ostringstream oss;
  oss << name << "(" << arg << ")";
  return oss.str();
}

// Helper: format a pass with benefit and a parenthesized int64 argument.
static std::string passWithBenefitAndArg(const char *name, int benefit,
                                         int64_t arg) {
  std::ostringstream oss;
  oss << name << "<" << benefit << ">(" << arg << ")";
  return oss.str();
}

static void addBaseTransformPasses(std::vector<std::string> &list,
                                   int64_t maxConstThreshold,
                                   int64_t whileUnrollThreshold) {
  // Canonicalization passes
  list.push_back("compare_op_canon<16>");
  list.push_back("transpose_transpose<16>");
  list.push_back("broadcast_in_dim_op_canon<16>");
  list.push_back("convert_op_canon<16>");
  list.push_back("dynamic_broadcast_in_dim_op_not_actually_dynamic<16>");
  list.push_back("chained_dynamic_broadcast_in_dim_canonicalization<16>");
  list.push_back("dynamic_broadcast_in_dim_all_dims_non_expanding<16>");
  list.push_back("noop_reduce_op_canon<16>");
  list.push_back("empty_reduce_op_canon<16>");
  list.push_back("dynamic_reshape_op_canon<16>");
  list.push_back("get_tuple_element_op_canon<16>");
  list.push_back("real_op_canon<16>");
  list.push_back("imag_op_canon<16>");
  list.push_back("conj_complex_negate<16>");
  list.push_back("get_dimension_size_op_canon<16>");
  list.push_back("reshape_op_canon<16>");
  list.push_back("merge_consecutive_reshapes<16>");
  list.push_back("transpose_is_reshape<16>");
  list.push_back("zero_extent_tensor_canon<16>");

  // CSE passes
  list.push_back("cse_broadcast_in_dim<16>");
  list.push_back("cse_slice<16>");
  list.push_back("cse_transpose<16>");
  list.push_back("cse_convert<16>");
  list.push_back("cse_dot_general<16>");
  list.push_back("cse_reshape<16>");
  list.push_back("cse_mul<16>");
  list.push_back("cse_div<16>");
  list.push_back("cse_add<16>");
  list.push_back("cse_subtract<16>");
  list.push_back("cse_min<16>");
  list.push_back("cse_max<16>");
  list.push_back("cse_neg<16>");
  list.push_back("cse_abs<16>");
  list.push_back("cse_concatenate<16>");
  list.push_back("cse_compare<16>");
  list.push_back("cse_select<16>");
  list.push_back("cse_real<16>");
  list.push_back("cse_imag<16>");
  list.push_back("cse_conj<16>");

  // Passes with max_constant_threshold
  list.push_back(passWithBenefitAndArg("concatenate_op_canon", 16,
                                       maxConstThreshold));
  list.push_back(
      passWithBenefitAndArg("select_op_canon", 16, maxConstThreshold));

  // Simplification passes
  list.push_back("add_simplify<16>");
  list.push_back("sub_simplify<16>");
  list.push_back("and_simplify<16>");
  list.push_back("max_simplify<16>");
  list.push_back("min_simplify<16>");
  list.push_back("or_simplify<16>");
  list.push_back("xor_simplify<16>");
  list.push_back("mul_simplify<16>");
  list.push_back("div_simplify<16>");
  list.push_back("rem_simplify<16>");
  list.push_back("pow_simplify<16>");
  list.push_back("simplify_extend<16>");
  list.push_back("simplify_wrap<16>");
  list.push_back("simplify_rotate<16>");
  list.push_back("extend_splat<16>");
  list.push_back("noop_slice<16>");
  list.push_back("noop_reverse<16>");
  list.push_back("slice_slice<16>");
  list.push_back("dynamic_slice_slice<16>");
  list.push_back("slice_dynamic_slice<16>");
  list.push_back("dynamic_slice_dynamic_slice<16>");
  list.push_back("shift_right_logical_simplify<16>");
  list.push_back("slice_simplify<16>");
  list.push_back("convert_simplify<16>");
  list.push_back("dynamic_slice_to_static<16>");
  list.push_back("dynamic_update_slice_elim<16>");
  list.push_back("concat_to_broadcast<16>");
  list.push_back("reduce_to_reshape<16>");
  list.push_back("broadcast_to_reshape<16>");
  list.push_back("slice_internal");

  // Passes with max_constant_threshold
  list.push_back(
      passWithBenefitAndArg("iota_simplify", 16, maxConstThreshold));
  list.push_back(passWithBenefitAndArg("broadcast_in_dim_simplify", 16,
                                       maxConstThreshold));

  // Fusion / rewrite passes
  list.push_back("convert_concat<1>");
  list.push_back("dynamic_update_to_concat<1>");
  list.push_back("slice_of_dynamic_update<1>");
  list.push_back("slice_elementwise<1>");
  list.push_back("dot_reshape_dot<1>");
  list.push_back("concat_fuse<1>");
  list.push_back("concat_push_binop_add<1>");
  list.push_back("concat_push_binop_mul<1>");
  list.push_back("reduce_concat<1>");
  list.push_back("slice_concat<1>");
  list.push_back("concat_slice<1>");
  list.push_back("select_op_used_within_if<1>");
  list.push_back("bin_broadcast_splat_add<1>");
  list.push_back("bin_broadcast_splat_subtract<1>");
  list.push_back("bin_broadcast_splat_div<1>");
  list.push_back("bin_broadcast_splat_mul<1>");
  list.push_back("dot_general_simplify<16>");
  list.push_back("transpose_simplify<16>");
  list.push_back("reshape_empty_broadcast<1>");
  list.push_back("broadcast_reshape<1>");
  list.push_back("transpose_dot_reorder<1>");
  list.push_back("dot_transpose<1>");
  list.push_back("transpose_convolution<1>");
  list.push_back("convolution_transpose<1>");
  list.push_back("convert_convert_float<1>");
  list.push_back("convert_convert_int<1>");
  list.push_back("reshape_iota<1>");
  list.push_back("broadcast_reduce<1>");
  list.push_back("slice_dot_general<1>");
  list.push_back("if_inline<1>");
  list.push_back("if_to_select<1>");
  list.push_back("divide_sqrt_to_multiply_rsqrt<16>");
  list.push_back("associative_binary_op_reordering<1>");
  list.push_back("transpose_broadcast_in_dim_to_broadcast_in_dim<16>");
  list.push_back("replace_neg_add_with_subtract");
  list.push_back("replace_subtract_neg_with_add");
  list.push_back("binop_const_simplify");
  list.push_back("not_select_simplify");
  list.push_back("common_compare_expression_rewrite");
  list.push_back("compare_select_simplify");
  list.push_back("while_simplify<1>(1)");
  list.push_back("if_remove_unused");
  list.push_back("transpose_reshape_to_broadcast");
  list.push_back("reshape_transpose_to_broadcast");
  list.push_back("reshape_broadcast");
  list.push_back("dus_dus");
  list.push_back("dus_dus_concat");
  list.push_back("abs_positive_simplify");
  list.push_back("transpose_elementwise_transpose");
  list.push_back("select_comp_iota_const_simplify<1>");
  list.push_back("sign_abs_simplify<1>");
  list.push_back("broadcastindim_is_reshape");
  list.push_back("reduce_window_wrap<1>");
  list.push_back("slice_reduce_window<1>");
  list.push_back("while_deadresult");
  list.push_back("while_idempotent_dus");
  list.push_back("while_dus");
  list.push_back("while_updatewithoutcorners");
  list.push_back("while_op_induction_replacement");
  list.push_back("dus_concat");
  list.push_back("dusdus_to_duspad");
  list.push_back("slice_dus_to_concat");
  list.push_back("sink_dus");
  list.push_back("hoist_slice");
  list.push_back("while_induction_reduction");
  list.push_back("slice_broadcast");
  list.push_back("associative_common_mul_op_reordering");
  list.push_back("slice_select_to_select_slice");
  list.push_back("slice_if");
  list.push_back("dus_to_i32");
  list.push_back("slice_extend");
  list.push_back("slice_of_updatewithoutcorners");
  list.push_back("concat_wrap");
  list.push_back("cse_updatewithoutcorners<16>");
  list.push_back("cse_extend<16>");
  list.push_back("cse_wrap<16>");
  list.push_back("cse_rotate<16>");
  list.push_back("concat_concat_axis_swap");
  list.push_back("concat_concat_to_dus");
  list.push_back("broadcast_iota_simplify");
  list.push_back("select_comp_iota_to_dus");
  list.push_back("compare_cleanup");
  list.push_back("broadcast_compare");
  list.push_back("not_compare");
  list.push_back("broadcast_iota");
  list.push_back("cse_iota");
  list.push_back("compare_iota_const_simplify");
  list.push_back("reshuffle_ands_compares");
  list.push_back("square_abs_simplify");
  list.push_back("divide_divide_simplify");
  list.push_back("concat_reshape_slice");
  list.push_back("full_reduce_reshape_or_transpose");
  list.push_back("concat_reshape_reduce");
  list.push_back("concat_elementwise");
  list.push_back("reduce_reduce");
  list.push_back("conj_real");
  list.push_back("select_broadcast_in_dim");
  list.push_back("if_op_lift_common_ops");
  list.push_back("involution_neg_simplify");
  list.push_back("involution_conj_simplify");
  list.push_back("involution_not_simplify");
  list.push_back("real_conj_simplify");
  list.push_back("real_convert_simplify");
  list.push_back("conj_complex_simplify");
  list.push_back("conj_convert_simplify");
  list.push_back("elementwise_complex_simplify");
  list.push_back("split_convolution_into_reverse_convolution");
  list.push_back("power_multiply_to_power");
  list.push_back("log_simplify");
  list.push_back("neg_mul_const_simplify");
  list.push_back("neg_div_const_simplify");
  list.push_back("reshape_deletions_broadcast_in_dim_simplify");
  list.push_back("reshape_insertions_broadcast_in_dim_simplify");
  list.push_back("dot_general_reshape");
  list.push_back("widen_wrap");
  list.push_back("widen_extend");
  list.push_back("elementwise_pad");
  list.push_back("compare_negate_const_simplify");
  list.push_back("select_simplify");
  list.push_back("concatenate_subtract_to_subtract_pad");
  list.push_back("concatenate_add_to_add_pad");
  list.push_back("concatenate_broadcast_in_dim");
  list.push_back("compare_abs");
  list.push_back("compare_convert");
  list.push_back("add_selects");
  list.push_back("self_subtract_to_convolution_like(0)");
  list.push_back("self_add_to_convolution_like(0)");
  list.push_back("self_mul_to_convolution_like(0)");
  list.push_back("subtract_multiply_const_to_add_mul_const");
  list.push_back("trivial_reduce_window_to_reduce_op");
  list.push_back("case_to_if");
  list.push_back("dot_general_add_distributive_simplify");
  list.push_back("dot_general_subtract_distributive_simplify");
  list.push_back("remove_no_ops_from_while_loop");
  list.push_back("while_is_copy_simplify");
  list.push_back("split_variadic_scatter_op");
  list.push_back("dynamic_slice_simplify");
  list.push_back(passWithArg("enzyme_hlo_unroll", whileUnrollThreshold));
  list.push_back("divide_negated_operands_simplify");
  list.push_back("multiply_negated_operands_simplify");
  list.push_back("factor_scalars_in_dot_general");
  list.push_back("reduce_mul_to_dot_general");
  list.push_back("dot_general_broadcast_in_dim");
  list.push_back("dot_general_broadcast_in_dim_sort_dims");
  list.push_back("dus_dynamic_slice_simplify");
  list.push_back("while_dus_dus_simplify");
  list.push_back("while_dus_ds_simplify");
  list.push_back("reshape_slice_reshape");
  list.push_back("dynamic_slice_elementwise");
  list.push_back("dot_general_remove_batch_dimensions");
  list.push_back("delete_dims_reduce");
  list.push_back("reduce_delete_dims");
  list.push_back("dot_general_insert_dim_contraction_simplification");
  list.push_back("fuse_reshape_collapse_or_expand_dims_into_reduce");
  list.push_back("split_reduce_add_mul_to_add_dot_general");
  list.push_back(passWithArg("recognize_from_constant", maxConstThreshold));
  list.push_back("extend_to_broadcast");
  list.push_back("reduce_max_min_mul_positive_scalar");
}

static void addStructuredTensorsSyrkPasses(std::vector<std::string> &list) {
  list.push_back("dot_general_to_syrk");
}

static void addStructuredTensorsPasses(std::vector<std::string> &list) {
  list.push_back("transpose_syrk_to_syrk");
  list.push_back("fuse_mul_into_syrk");
  list.push_back("fuse_add_into_syrk");
  list.push_back("dot_general_only_diagonal_access");
  list.push_back("transpose_symmetric_simplify");
  list.push_back("syrk_simplify_output_uplo");
}

static void addScatterGatherPasses(std::vector<std::string> &list,
                                   int64_t maxConstThreshold) {
  // scatter patterns
  list.push_back("scatter_op_canon<16>");
  list.push_back("scatter_to_dynamic_update_slice<1>");
  list.push_back("scatter_multiply_simplify");
  list.push_back("scatter_sub_simplify");
  list.push_back("scatter_add_simplify");
  list.push_back("scatter_div_simplify");
  list.push_back("unary_elementwise_scatter_simplify");
  list.push_back("scatter_indices_are_unique");
  list.push_back("split_complex_scatter");
  list.push_back("split_complex_gather");
  // const prop patterns
  list.push_back("scatter_update_computation_const_prop");
  // gather patterns
  list.push_back("dynamic_gather_op_is_not_dynamic<16>");
  list.push_back("gather_op_canon<16>");
  list.push_back("gather_elementwise");
  list.push_back("elementwise_gather");
  list.push_back("gather_of_scatter_simplify");
  // const prop patterns
  list.push_back("gather_const_prop");
  list.push_back(passWithArg("scatter_const_fold", maxConstThreshold));
  list.push_back("cse_gather");
  list.push_back("cse_scatter");
}

static void addSliceToBatchPasses(std::vector<std::string> &list) {
  list.push_back("dot_general_slice_to_batch");
  list.push_back("gather_slice_to_batch");
  list.push_back("iota_slice_to_batch");
  list.push_back("reduce_slice_to_batch");
  list.push_back("sort_slice_to_batch");
  list.push_back("transpose_slice_to_batch");
  list.push_back("broadcastindim_slice_to_batch");
  list.push_back("reducewindow_slice_to_batch");
  list.push_back("elementwise_slice_to_batch");
  list.push_back("convolution_slice_to_batch");
}

static void addReduceSliceFusionPasses(std::vector<std::string> &list) {
  list.push_back("add_reduce_slice_fusion");
  list.push_back("mul_reduce_slice_fusion");
  list.push_back("min_reduce_slice_fusion");
  list.push_back("max_reduce_slice_fusion");
  list.push_back("and_reduce_slice_fusion");
  list.push_back("xor_reduce_slice_fusion");
  list.push_back("or_reduce_slice_fusion");
}

static void addConcatToBatchPasses(std::vector<std::string> &list) {
  list.push_back("concat_insert_dim_dot_general");
  list.push_back("concat_insert_dim_gather");
  list.push_back("concat_insert_dim_iota");
  list.push_back("concat_insert_dim_reduce");
  list.push_back("concat_insert_dim_sort");
  list.push_back("concat_insert_dim_reduce_window");
  list.push_back("concat_insert_dim_elementwise");
  list.push_back("concat_insert_dim_convolution");
}

static void addLoopRaisingPasses(std::vector<std::string> &list) {
  list.push_back("greedy_while_loop_batch_fission");
  list.push_back("while_elementwise_reduction_to_reduce");
  list.push_back("remove_loop_carried_dependencies_from_while_load_operations");
}

static void addLICMPasses(std::vector<std::string> &list) {
  list.push_back("dus_licm(0)");
  list.push_back("slice_licm(0)");
  list.push_back("elementwise_licm(0)");
  list.push_back("concatenate_licm(0)");
  list.push_back("while_licm<1>(1)");
  list.push_back("transpose_licm(0)");
  list.push_back("broadcastindim_licm(0)");
  list.push_back("reshape_licm(0)");
  list.push_back("dot_general_licm(0)");
  list.push_back("reduce_licm(0)");
  list.push_back("reduce_window_licm(0)");
  list.push_back("reverse_licm(0)");
  list.push_back("convolution_licm(0)");
  list.push_back("dynamic_slice_licm(0)");
  list.push_back("scatter_licm(0)");
  list.push_back("gather_licm(0)");
  list.push_back("iota_licm(0)");
  list.push_back("rotate_licm(0)");
  list.push_back("wrap_licm(0)");
  list.push_back("extend_licm(0)");
}

static void addPadPasses(std::vector<std::string> &list,
                         int64_t maxConstThreshold, bool enableLICM) {
  list.push_back("extend_pad");
  list.push_back("dus_pad");
  list.push_back("cse_pad<16>");
  list.push_back(
      passWithBenefitAndArg("pad_simplify", 16, maxConstThreshold));
  list.push_back("select_pad_to_dus<1>");
  list.push_back("and_pad_pad<1>");
  list.push_back("negative_pad_to_slice<16>");
  list.push_back("slice_pad<1>");
  list.push_back("pad_reshape_pad<1>");
  list.push_back("pad_pad<1>");
  list.push_back("add_pad_pad_to_concat<1>");
  list.push_back("concat_pad<1>");
  list.push_back("reduce_pad<1>");
  list.push_back("broadcast_pad<1>");
  list.push_back("zero_product_reshape_pad<1>");
  list.push_back("mul_zero_pad<1>");
  list.push_back("div_zero_pad<1>");
  list.push_back("binop_const_reshape_pad<1>");
  list.push_back("binop_pad_to_concat_add<1>");
  list.push_back("binop_pad_to_concat_mul<1>");
  list.push_back("binop_const_pad_add<1>");
  list.push_back("binop_const_pad_subtract<1>");
  list.push_back("binop_const_pad_mul<1>");
  list.push_back("binop_const_pad_div<1>");
  list.push_back("binop_binop_pad_pad_add<1>");
  list.push_back("binop_binop_pad_pad_mul<1>");
  list.push_back("binop_pad_pad_add<1>");
  list.push_back("binop_pad_pad_subtract<1>");
  list.push_back("binop_pad_pad_mul<1>");
  list.push_back("binop_pad_pad_div<1>");
  list.push_back("binop_pad_pad_min<1>");
  list.push_back("binop_pad_pad_max<1>");
  list.push_back("unary_pad_push_convert<1>");
  list.push_back("unary_pad_push_tanh<1>");
  list.push_back("unary_pad_push_exp<1>");
  list.push_back("concat_to_pad<1>");
  list.push_back("while_pad_induction_reduction");
  list.push_back("pad_concat_to_concat_pad");
  list.push_back("rotate_pad");
  list.push_back("concat_multipad");
  list.push_back("speculate_if_pad_to_select");
  list.push_back("dus_to_dynamic_pad");
  list.push_back("dynamic_pad_to_pad");

  if (enableLICM) {
    list.push_back("pad_licm(0)");
  }
}

static void addConstPropPasses(std::vector<std::string> &list,
                               int64_t maxConstThreshold) {
  // Unary constant propagation
  list.push_back("chlo_inf_const_prop<16>");
  list.push_back("gamma_const_prop<16>");
  list.push_back("abs_const_prop<16>");
  list.push_back("log_const_prop<1>");
  list.push_back("log_plus_one_const_prop<1>");
  list.push_back("is_finite_const_prop");
  list.push_back("not_const_prop");
  list.push_back("neg_const_prop");
  list.push_back("sqrt_const_prop");
  list.push_back("rsqrt_const_prop");
  list.push_back("cos_const_prop");
  list.push_back("sin_const_prop");
  list.push_back("exp_const_prop");
  list.push_back("expm1_const_prop");
  list.push_back("tanh_const_prop");
  list.push_back("logistic_const_prop");
  list.push_back("conj_const_prop");
  list.push_back("ceil_const_prop");
  list.push_back("cbrt_const_prop");
  list.push_back("real_const_prop");
  list.push_back("imag_const_prop");
  list.push_back("round_const_prop");
  list.push_back("round_nearest_even_const_prop");
  list.push_back("sign_const_prop");
  list.push_back("floor_const_prop");
  list.push_back("tan_const_prop");

  // Binary constant propagation
  list.push_back("add_const_prop");
  list.push_back("and_const_prop");
  list.push_back("atan2_const_prop");
  list.push_back("complex_const_prop");
  list.push_back("div_const_prop");
  list.push_back("max_const_prop");
  list.push_back("min_const_prop");
  list.push_back("mul_const_prop");
  list.push_back("or_const_prop");
  list.push_back("pow_const_prop");
  list.push_back("rem_const_prop");
  list.push_back("sub_const_prop");
  list.push_back("xor_const_prop");

  // Other constant propagations
  list.push_back(
      passWithBenefitAndArg("concat_const_prop", 1, maxConstThreshold));
  list.push_back(
      passWithArg("dynamic_update_slice_const_prop", maxConstThreshold));
  list.push_back("clamp_const_prop");
}

static void addReshapePropagateUpPasses(std::vector<std::string> &list,
                                        bool aggressive) {
  list.push_back("reshape_concat");
  list.push_back("reshape_dus");
  list.push_back("dot_reshape_pad<1>");
  list.push_back("pad_dot_general<1>(0)");
  list.push_back("reshape_pad");
  list.push_back("reshape_wrap");
  list.push_back("reshape_rotate");
  list.push_back("reshape_extend");
  list.push_back("delete_dims_broadcast");

  if (aggressive) {
    list.push_back("reshape_slice(0)");
    list.push_back("reshape_elementwise(0)");
    list.push_back("reshape_dynamic_slice(0)");
  } else {
    list.push_back("reshape_slice(1)");
    list.push_back("reshape_elementwise(1)");
    list.push_back("reshape_dynamic_slice(1)");
  }
}

static void addReshapePropagateDownPasses(std::vector<std::string> &list,
                                          bool aggressive) {
  list.push_back("concat_appending_reshape");
  list.push_back("slice_reshape");
  list.push_back("slice_reshape_slice<1>");
  list.push_back("dynamic_slice_reshape_slice<1>");
  list.push_back("slice_reshape_dynamic_slice<1>");
  list.push_back("dynamic_slice_reshape_dynamic_slice<1>");
  list.push_back("slice_reshape_concat<1>");
  list.push_back("slice_reshape_elementwise<1>");
  list.push_back("slice_reshape_dot_general<1>");
  list.push_back("slice_reshape_pad<1>");
  list.push_back("elementwise_reshape_like");

  if (aggressive) {
    list.push_back("reshape_elementwise_only_fusible(0)");
  } else {
    list.push_back("reshape_elementwise_only_fusible(1)");
  }
}

static void addTransposePropagateUpPasses(std::vector<std::string> &list,
                                          bool aggressive) {
  list.push_back("transpose_select");
  list.push_back("transpose_while");
  list.push_back("transpose_slice");
  list.push_back("transpose_like_broadcast_slice");
  list.push_back("transpose_concat");
  list.push_back("transpose_iota");
  list.push_back("transpose_reduce");
  list.push_back("transpose_reduce_window");
  list.push_back("transpose_dus");
  list.push_back("transpose_pad<1>");
  list.push_back("transpose_einsum<1>");
  list.push_back("transpose_wrap");
  list.push_back("transpose_extend");
  list.push_back("transpose_rotate");
  list.push_back("transpose_dynamic_slice");
  list.push_back("transpose_like_broadcast_dynamic_slice");
  list.push_back("transpose_reverse");
  list.push_back("transpose_batch_norm_training");
  list.push_back("transpose_batch_norm_inference");
  list.push_back("transpose_batch_norm_grad");
  list.push_back("transpose_if");
  list.push_back("transpose_fft");
  list.push_back("transpose_reshape");

  if (aggressive) {
    list.push_back("transpose_elementwise(0)");
    list.push_back("transpose_like_broadcast_elementwise(0)");
  } else {
    list.push_back("transpose_elementwise(1)");
    list.push_back("transpose_like_broadcast_elementwise(1)");
  }
}

static void addTransposePropagateDownPasses(std::vector<std::string> &list) {
  list.push_back("reorder_elementwise_and_shape_op<16>");
  list.push_back("elementwise_all_transpose_operands_simplify");
  list.push_back("slice_transpose");
  list.push_back("dynamic_slice_transpose");
  list.push_back("einsum_transpose<1>");
  list.push_back("slice_reshape_transpose<1>");
  list.push_back("reduce_transpose_simplify");
  list.push_back("reverse_transpose");
  list.push_back("transpose_all_users_slice");
}

static void addNoNanPasses(std::vector<std::string> &list, bool noNan) {
  const char *arg = noNan ? "(1)" : "(0)";
  list.push_back(std::string("no_nan_compare_simplify") + arg);
  list.push_back(std::string("no_nan_self_sub_simplify") + arg);
  list.push_back(std::string("no_nan_add_sub_simplify") + arg);
  list.push_back(std::string("no_nan_mul_simplify") + arg);
  list.push_back(std::string("no_nan_div_simplify") + arg);
}

static void addAllFinitePasses(std::vector<std::string> &list) {
  list.push_back("all_finite_is_finite");
  list.push_back("all_finite_is_inf");
  list.push_back("all_finite_is_pos_inf");
  list.push_back("all_finite_is_neg_inf");
}

static void addRecognizeCommsPasses(std::vector<std::string> &list) {
  list.push_back("recognize_extend");
  list.push_back("recognize_wrap");
  list.push_back("recognize_rotate");
  list.push_back("recognize_updatewithoutcorners");
  list.push_back("dusdus_to_dusextend");
}

static void addLowerCommsPasses(std::vector<std::string> &list) {
  list.push_back("lower_extend");
  list.push_back("lower_wrap");
  list.push_back("lower_rotate");
  list.push_back("lower_updatewithoutcorners");
}

// Join a vector of strings with a separator.
static std::string joinWith(const std::vector<std::string> &vec,
                            const char *sep) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i > 0)
      oss << sep;
    oss << vec[i];
  }
  return oss.str();
}

// Allocate a C string copy (caller must free).
static char *strdupCopy(const std::string &s) {
  char *result = (char *)malloc(s.size() + 1);
  if (result) {
    memcpy(result, s.c_str(), s.size() + 1);
  }
  return result;
}

} // namespace

void enzymexlaGetTransformPassesList(
    const EnzymeXLATransformPassesOptions *options, char **mainPasses,
    char **lowerPasses) {
  std::vector<std::string> list;
  int64_t maxConst = options->max_constant_threshold;
  int64_t whileUnroll = options->while_unroll_threshold;
  bool aggressive = options->aggressive_propagation;

  // Base passes
  addBaseTransformPasses(list, maxConst, whileUnroll);

  // Structured tensors detection (syrk) — only when not sharded
  if (!options->is_sharded && options->raise_shlo_to_blas_lapack &&
      options->enable_structured_tensors_detection_passes) {
    addStructuredTensorsSyrkPasses(list);
  }

  // Structured tensors passes
  if (options->enable_structured_tensors_passes) {
    addStructuredTensorsPasses(list);
  }

  // Scatter/gather
  if (options->enable_scatter_gather_optimization_passes) {
    addScatterGatherPasses(list, maxConst);
  }

  // Diagonal tensor rewrite (requires both scatter/gather and structured
  // tensors)
  if (options->enable_scatter_gather_optimization_passes &&
      options->enable_structured_tensors_passes) {
    list.push_back("diagonal_tensor_dot_general_rewrite");
  }

  // Slice to batch
  if (options->enable_slice_to_batch_passes) {
    addSliceToBatchPasses(list);
  }

  // Reduce slice fusion
  if (options->enable_reduce_slice_fusion_passes) {
    addReduceSliceFusionPasses(list);
  }

  // Concat to batch
  if (options->enable_concat_to_batch_passes) {
    addConcatToBatchPasses(list);
  }

  // Loop raising
  if (options->enable_loop_raising_passes) {
    addLoopRaisingPasses(list);
  }

  // LICM
  if (options->enable_licm_optimization_passes) {
    addLICMPasses(list);
  }

  // Pad passes
  if (options->enable_pad_optimization_passes) {
    addPadPasses(list, maxConst, options->enable_licm_optimization_passes);
  }

  // Constant propagation
  addConstPropPasses(list, maxConst);

  // Optional runtime flags
  if (options->dus_slice_simplify) {
    list.push_back("dus_slice_simplify");
  }
  if (options->sum_to_reducewindow) {
    list.push_back("sum_to_reducewindow");
  }
  if (options->sum_to_conv) {
    list.push_back("sum_to_conv(0)");
  }
  if (options->aggressive_sum_to_conv) {
    list.push_back("sum_to_conv(1)");
  }
  if (options->while_concat) {
    list.push_back("while_concat");
    list.push_back("while_wrap");
    list.push_back("while_extend");
  }
  if (options->dus_to_concat) {
    list.push_back("dus_to_concat");
  }

  // Reshape propagation
  if (options->reshape_propagate == ENZYMEXLA_PROPAGATE_UP) {
    addReshapePropagateUpPasses(list, aggressive);
  } else if (options->reshape_propagate == ENZYMEXLA_PROPAGATE_DOWN) {
    addReshapePropagateDownPasses(list, aggressive);
  }

  // Transpose propagation
  if (options->transpose_propagate == ENZYMEXLA_PROPAGATE_UP) {
    addTransposePropagateUpPasses(list, aggressive);
  } else if (options->transpose_propagate == ENZYMEXLA_PROPAGATE_DOWN) {
    addTransposePropagateDownPasses(list);
  }

  // no_nan passes (always added, parameterized by flag)
  addNoNanPasses(list, options->no_nan);

  // all_finite
  if (options->all_finite) {
    addAllFinitePasses(list);
  }

  // Build the lower_transform_passes as a copy of list at this point
  std::vector<std::string> lowerList = list;

  // Recognize comms (added to main list only)
  if (options->recognize_comms) {
    addRecognizeCommsPasses(list);
  }

  // Lower comms (added to lower list only)
  if (options->lower_comms) {
    addLowerCommsPasses(lowerList);
  }

  // Output
  std::string mainStr = joinWith(list, ";");
  std::string lowerStr = joinWith(lowerList, ";");

  *mainPasses = strdupCopy(mainStr);
  *lowerPasses = strdupCopy(lowerStr);
}

void enzymexlaFreeTransformPassesList(char *passes) { free(passes); }
