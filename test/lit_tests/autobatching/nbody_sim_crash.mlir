// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(inline{default-pipeline=canonicalize max-iterations=4},canonicalize,cse,canonicalize,enzyme-hlo-generate-td{patterns=transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;mul_simplify<16>;div_simplify<16>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;broadcast_reshape<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;convert_convert_int<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;replace_neg_add_with_subtract;replace_subtract_neg_with_add;binop_const_simplify;not_select_simplify;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;sign_abs_simplify<1>;slice_broadcast;associative_common_mul_op_reordering;slice_select_to_select_slice;slice_if;dus_to_i32;slice_extend;cse_rotate<16>;concat_concat_axis_swap;concat_concat_to_dus;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;conj_real;select_broadcast_in_dim;if_op_lift_common_ops;involution_neg_simplify;involution_conj_simplify;involution_not_simplify;real_conj_simplify;conj_complex_simplify;split_convolution_into_reverse_convolution;power_multiply_to_power;log_simplify;neg_mul_const_simplify;neg_div_const_simplify;reshape_deletions_broadcast_in_dim_simplify;reshape_insertions_broadcast_in_dim_simplify;dot_general_reshape;widen_wrap;widen_extend;elementwise_pad;compare_negate_const_simplify;select_simplify;concatenate_subtract_to_subtract_pad;concatenate_broadcast_in_dim;compare_abs;compare_convert;add_selects;transpose_symmetric_simplify;divide_negated_operands_simplify;multiply_negated_operands_simplify;transpose_syrk_to_syrk;fuse_mul_into_syrk;fuse_add_into_syrk;factor_scalars_in_dot_general;reduce_mul_to_dot_general;dot_general_broadcast_in_dim;dot_general_broadcast_in_dim_sort_dims;dus_dynamic_slice_simplify;while_dus_ds_simplify;dot_general_to_syrk;add_reduce_slice_fusion;mul_reduce_slice_fusion;min_reduce_slice_fusion;max_reduce_slice_fusion;concat_insert_dim_dot_general;concat_insert_dim_gather;concat_insert_dim_iota;concat_insert_dim_reduce;reduce_slice_to_batch;sort_slice_to_batch;transpose_slice_to_batch;broadcastindim_slice_to_batch;reducewindow_slice_to_batch;elementwise_slice_to_batch;convolution_slice_to_batch;greedy_while_loop_batch_fission;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);reshape_dynamic_slice(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_batch_norm_training;transpose_batch_norm_inference;transpose_batch_norm_grad;transpose_if;transpose_fft;transpose_reshape;transpose_elementwise(1);no_nan_compare_simplify(0);no_nan_self_sub_simplify(0);no_nan_add_sub_simplify(0);no_nan_mul_simplify(0);no_nan_div_simplify(0);recognize_extend;recognize_wrap;recognize_rotate},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_compute... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<1024x3xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}, %arg1: tensor<1024xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 2 : i32}) -> (tensor<1024x1024xf32>, tensor<1024x3xf32>, tensor<1024xf32>) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1024x3xf32>) -> tensor<3x1024xf32>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %2 = stablehlo.convert %c : tensor<i64>
    %c_0 = stablehlo.constant dense<1024> : tensor<i64>
    %3 = stablehlo.convert %c_0 : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %4 = stablehlo.convert %c_1 : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %5 = stablehlo.convert %c_2 : tensor<i64>
    %c_3 = stablehlo.constant dense<1024> : tensor<i64>
    %6 = stablehlo.convert %c_3 : tensor<i64>
    %cst_4 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %7 = stablehlo.convert %cst_4 : tensor<f32>
    // CHECK: %8 = stablehlo.compare  EQ, %6, %7 : (tensor<1024x1024x1x1xi64>, tensor<1024x1024x1x1xi64>) -> tensor<1024x1024x1x1xi1>
    %8:9 = stablehlo.while(%iterArg = %5, %iterArg_5 = %2, %iterArg_6 = %3, %iterArg_7 = %4, %iterArg_8 = %0, %iterArg_9 = %6, %iterArg_10 = %cst, %iterArg_11 = %7, %iterArg_12 = %1) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x1024xf32>, tensor<i64>, tensor<1024x1024xf32>, tensor<f32>, tensor<1024xf32> attributes {enzyme.disable_mincut}
    cond {
      %12 = stablehlo.subtract %iterArg_6, %iterArg_5 : tensor<i64>
      %13 = stablehlo.divide %12, %iterArg_7 : tensor<i64>
      %c_13 = stablehlo.constant dense<1> : tensor<i64>
      %14 = stablehlo.convert %c_13 : tensor<i64>
      %15 = stablehlo.add %13, %14 : tensor<i64>
      %16 = stablehlo.compare  LT, %iterArg, %15 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %16 : tensor<i1>
    } do {
      %12 = stablehlo.multiply %iterArg, %iterArg_7 : tensor<i64>
      %13 = stablehlo.add %iterArg_5, %12 : tensor<i64>
      %c_13 = stablehlo.constant dense<1> : tensor<i64>
      %14 = stablehlo.convert %c_13 : tensor<i64>
      %15 = stablehlo.add %iterArg, %14 : tensor<i64>
      %c_14 = stablehlo.constant dense<1> : tensor<i64>
      %16 = stablehlo.convert %c_14 : tensor<i64>
      %c_15 = stablehlo.constant dense<1> : tensor<i64>
      %17 = stablehlo.convert %c_15 : tensor<i64>
      %c_16 = stablehlo.constant dense<0> : tensor<i64>
      %18 = stablehlo.convert %c_16 : tensor<i64>
      %19:9 = stablehlo.while(%iterArg_17 = %18, %iterArg_18 = %iterArg_9, %iterArg_19 = %16, %iterArg_20 = %17, %iterArg_21 = %iterArg_8, %iterArg_22 = %iterArg_10, %iterArg_23 = %iterArg_11, %iterArg_24 = %13, %iterArg_25 = %iterArg_12) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x1024xf32>, tensor<1024x1024xf32>, tensor<f32>, tensor<i64>, tensor<1024xf32> attributes {enzyme.disable_mincut}
      cond {
        %20 = stablehlo.subtract %iterArg_18, %iterArg_19 : tensor<i64>
        %21 = stablehlo.divide %20, %iterArg_20 : tensor<i64>
        %c_26 = stablehlo.constant dense<1> : tensor<i64>
        %22 = stablehlo.convert %c_26 : tensor<i64>
        %23 = stablehlo.add %21, %22 : tensor<i64>
        %24 = stablehlo.compare  LT, %iterArg_17, %23 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %24 : tensor<i1>
      } do {
        %20 = stablehlo.multiply %iterArg_17, %iterArg_20 : tensor<i64>
        %21 = stablehlo.add %iterArg_19, %20 : tensor<i64>
        %c_26 = stablehlo.constant dense<1> : tensor<i64>
        %22 = stablehlo.convert %c_26 : tensor<i64>
        %23 = stablehlo.add %iterArg_17, %22 : tensor<i64>
        %c_27 = stablehlo.constant dense<1> : tensor<i64>
        %24 = stablehlo.convert %c_27 : tensor<i64>
        %25 = stablehlo.convert %24 : (tensor<i64>) -> tensor<i32>
        %c_28 = stablehlo.constant dense<1> : tensor<i32>
        %26 = stablehlo.convert %c_28 : tensor<i32>
        %27 = stablehlo.subtract %25, %26 : tensor<i32>
        %28 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_29 = stablehlo.constant dense<1> : tensor<i32>
        %29 = stablehlo.convert %c_29 : tensor<i32>
        %30 = stablehlo.subtract %28, %29 : tensor<i32>
        %31 = stablehlo.dynamic_slice %iterArg_21, %27, %30, sizes = [1, 1] : (tensor<3x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %32 = stablehlo.transpose %31, dims = [1, 0] : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %33 = stablehlo.reshape %32 : (tensor<1x1xf32>) -> tensor<f32>
        %34 = stablehlo.transpose %33, dims = [] : (tensor<f32>) -> tensor<f32>
        %c_30 = stablehlo.constant dense<1> : tensor<i64>
        %35 = stablehlo.convert %c_30 : tensor<i64>
        %36 = stablehlo.convert %35 : (tensor<i64>) -> tensor<i32>
        %c_31 = stablehlo.constant dense<1> : tensor<i32>
        %37 = stablehlo.convert %c_31 : tensor<i32>
        %38 = stablehlo.subtract %36, %37 : tensor<i32>
        %39 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_32 = stablehlo.constant dense<1> : tensor<i32>
        %40 = stablehlo.convert %c_32 : tensor<i32>
        %41 = stablehlo.subtract %39, %40 : tensor<i32>
        %42 = stablehlo.dynamic_slice %iterArg_21, %38, %41, sizes = [1, 1] : (tensor<3x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %43 = stablehlo.transpose %42, dims = [1, 0] : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %44 = stablehlo.reshape %43 : (tensor<1x1xf32>) -> tensor<f32>
        %45 = stablehlo.transpose %44, dims = [] : (tensor<f32>) -> tensor<f32>
        %46 = stablehlo.subtract %34, %45 : tensor<f32>
        %c_33 = stablehlo.constant dense<2> : tensor<i64>
        %47 = stablehlo.convert %c_33 : tensor<i64>
        %48 = stablehlo.convert %47 : (tensor<i64>) -> tensor<i32>
        %c_34 = stablehlo.constant dense<1> : tensor<i32>
        %49 = stablehlo.convert %c_34 : tensor<i32>
        %50 = stablehlo.subtract %48, %49 : tensor<i32>
        %51 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_35 = stablehlo.constant dense<1> : tensor<i32>
        %52 = stablehlo.convert %c_35 : tensor<i32>
        %53 = stablehlo.subtract %51, %52 : tensor<i32>
        %54 = stablehlo.dynamic_slice %iterArg_21, %50, %53, sizes = [1, 1] : (tensor<3x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %55 = stablehlo.transpose %54, dims = [1, 0] : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %56 = stablehlo.reshape %55 : (tensor<1x1xf32>) -> tensor<f32>
        %57 = stablehlo.transpose %56, dims = [] : (tensor<f32>) -> tensor<f32>
        %c_36 = stablehlo.constant dense<2> : tensor<i64>
        %58 = stablehlo.convert %c_36 : tensor<i64>
        %59 = stablehlo.convert %58 : (tensor<i64>) -> tensor<i32>
        %c_37 = stablehlo.constant dense<1> : tensor<i32>
        %60 = stablehlo.convert %c_37 : tensor<i32>
        %61 = stablehlo.subtract %59, %60 : tensor<i32>
        %62 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_38 = stablehlo.constant dense<1> : tensor<i32>
        %63 = stablehlo.convert %c_38 : tensor<i32>
        %64 = stablehlo.subtract %62, %63 : tensor<i32>
        %65 = stablehlo.dynamic_slice %iterArg_21, %61, %64, sizes = [1, 1] : (tensor<3x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %66 = stablehlo.transpose %65, dims = [1, 0] : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %67 = stablehlo.reshape %66 : (tensor<1x1xf32>) -> tensor<f32>
        %68 = stablehlo.transpose %67, dims = [] : (tensor<f32>) -> tensor<f32>
        %69 = stablehlo.subtract %57, %68 : tensor<f32>
        %c_39 = stablehlo.constant dense<3> : tensor<i64>
        %70 = stablehlo.convert %c_39 : tensor<i64>
        %71 = stablehlo.convert %70 : (tensor<i64>) -> tensor<i32>
        %c_40 = stablehlo.constant dense<1> : tensor<i32>
        %72 = stablehlo.convert %c_40 : tensor<i32>
        %73 = stablehlo.subtract %71, %72 : tensor<i32>
        %74 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_41 = stablehlo.constant dense<1> : tensor<i32>
        %75 = stablehlo.convert %c_41 : tensor<i32>
        %76 = stablehlo.subtract %74, %75 : tensor<i32>
        %77 = stablehlo.dynamic_slice %iterArg_21, %73, %76, sizes = [1, 1] : (tensor<3x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %78 = stablehlo.transpose %77, dims = [1, 0] : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %79 = stablehlo.reshape %78 : (tensor<1x1xf32>) -> tensor<f32>
        %80 = stablehlo.transpose %79, dims = [] : (tensor<f32>) -> tensor<f32>
        %c_42 = stablehlo.constant dense<3> : tensor<i64>
        %81 = stablehlo.convert %c_42 : tensor<i64>
        %82 = stablehlo.convert %81 : (tensor<i64>) -> tensor<i32>
        %c_43 = stablehlo.constant dense<1> : tensor<i32>
        %83 = stablehlo.convert %c_43 : tensor<i32>
        %84 = stablehlo.subtract %82, %83 : tensor<i32>
        %85 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_44 = stablehlo.constant dense<1> : tensor<i32>
        %86 = stablehlo.convert %c_44 : tensor<i32>
        %87 = stablehlo.subtract %85, %86 : tensor<i32>
        %88 = stablehlo.dynamic_slice %iterArg_21, %84, %87, sizes = [1, 1] : (tensor<3x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %89 = stablehlo.transpose %88, dims = [1, 0] : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %90 = stablehlo.reshape %89 : (tensor<1x1xf32>) -> tensor<f32>
        %91 = stablehlo.transpose %90, dims = [] : (tensor<f32>) -> tensor<f32>
        %92 = stablehlo.subtract %80, %91 : tensor<f32>
        %93 = stablehlo.compare  EQ, %iterArg_24, %21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %94 = stablehlo.multiply %46, %46 : tensor<f32>
        %95 = stablehlo.multiply %69, %69 : tensor<f32>
        %96 = stablehlo.multiply %92, %92 : tensor<f32>
        %97 = stablehlo.add %94, %95 : tensor<f32>
        %98 = stablehlo.add %97, %96 : tensor<f32>
        %cst_45 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %99 = stablehlo.convert %cst_45 : tensor<f32>
        %100 = stablehlo.divide %99, %98 : tensor<f32>
        %101 = stablehlo.select %93, %46, %100 : tensor<i1>, tensor<f32>
        %102 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_46 = stablehlo.constant dense<1> : tensor<i32>
        %103 = stablehlo.convert %c_46 : tensor<i32>
        %104 = stablehlo.subtract %102, %103 : tensor<i32>
        %105 = stablehlo.dynamic_slice %iterArg_25, %104, sizes = [1] : (tensor<1024xf32>, tensor<i32>) -> tensor<1xf32>
        %106 = stablehlo.transpose %105, dims = [0] : (tensor<1xf32>) -> tensor<1xf32>
        %107 = stablehlo.reshape %106 : (tensor<1xf32>) -> tensor<f32>
        %108 = stablehlo.transpose %107, dims = [] : (tensor<f32>) -> tensor<f32>
        %109 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_47 = stablehlo.constant dense<1> : tensor<i32>
        %110 = stablehlo.convert %c_47 : tensor<i32>
        %111 = stablehlo.subtract %109, %110 : tensor<i32>
        %112 = stablehlo.dynamic_slice %iterArg_25, %111, sizes = [1] : (tensor<1024xf32>, tensor<i32>) -> tensor<1xf32>
        %113 = stablehlo.transpose %112, dims = [0] : (tensor<1xf32>) -> tensor<1xf32>
        %114 = stablehlo.reshape %113 : (tensor<1xf32>) -> tensor<f32>
        %115 = stablehlo.transpose %114, dims = [] : (tensor<f32>) -> tensor<f32>
        %116 = stablehlo.multiply %iterArg_23, %108 : tensor<f32>
        %117 = stablehlo.multiply %116, %115 : tensor<f32>
        %118 = stablehlo.multiply %117, %101 : tensor<f32>
        %119 = stablehlo.multiply %118, %46 : tensor<f32>
        %120 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_48 = stablehlo.constant dense<1> : tensor<i32>
        %121 = stablehlo.convert %c_48 : tensor<i32>
        %122 = stablehlo.subtract %120, %121 : tensor<i32>
        %123 = stablehlo.dynamic_slice %iterArg_25, %122, sizes = [1] : (tensor<1024xf32>, tensor<i32>) -> tensor<1xf32>
        %124 = stablehlo.transpose %123, dims = [0] : (tensor<1xf32>) -> tensor<1xf32>
        %125 = stablehlo.reshape %124 : (tensor<1xf32>) -> tensor<f32>
        %126 = stablehlo.transpose %125, dims = [] : (tensor<f32>) -> tensor<f32>
        %127 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_49 = stablehlo.constant dense<1> : tensor<i32>
        %128 = stablehlo.convert %c_49 : tensor<i32>
        %129 = stablehlo.subtract %127, %128 : tensor<i32>
        %130 = stablehlo.dynamic_slice %iterArg_25, %129, sizes = [1] : (tensor<1024xf32>, tensor<i32>) -> tensor<1xf32>
        %131 = stablehlo.transpose %130, dims = [0] : (tensor<1xf32>) -> tensor<1xf32>
        %132 = stablehlo.reshape %131 : (tensor<1xf32>) -> tensor<f32>
        %133 = stablehlo.transpose %132, dims = [] : (tensor<f32>) -> tensor<f32>
        %134 = stablehlo.multiply %iterArg_23, %126 : tensor<f32>
        %135 = stablehlo.multiply %134, %133 : tensor<f32>
        %136 = stablehlo.multiply %135, %101 : tensor<f32>
        %137 = stablehlo.multiply %136, %69 : tensor<f32>
        %138 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_50 = stablehlo.constant dense<1> : tensor<i32>
        %139 = stablehlo.convert %c_50 : tensor<i32>
        %140 = stablehlo.subtract %138, %139 : tensor<i32>
        %141 = stablehlo.dynamic_slice %iterArg_25, %140, sizes = [1] : (tensor<1024xf32>, tensor<i32>) -> tensor<1xf32>
        %142 = stablehlo.transpose %141, dims = [0] : (tensor<1xf32>) -> tensor<1xf32>
        %143 = stablehlo.reshape %142 : (tensor<1xf32>) -> tensor<f32>
        %144 = stablehlo.transpose %143, dims = [] : (tensor<f32>) -> tensor<f32>
        %145 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_51 = stablehlo.constant dense<1> : tensor<i32>
        %146 = stablehlo.convert %c_51 : tensor<i32>
        %147 = stablehlo.subtract %145, %146 : tensor<i32>
        %148 = stablehlo.dynamic_slice %iterArg_25, %147, sizes = [1] : (tensor<1024xf32>, tensor<i32>) -> tensor<1xf32>
        %149 = stablehlo.transpose %148, dims = [0] : (tensor<1xf32>) -> tensor<1xf32>
        %150 = stablehlo.reshape %149 : (tensor<1xf32>) -> tensor<f32>
        %151 = stablehlo.transpose %150, dims = [] : (tensor<f32>) -> tensor<f32>
        %152 = stablehlo.multiply %iterArg_23, %144 : tensor<f32>
        %153 = stablehlo.multiply %152, %151 : tensor<f32>
        %154 = stablehlo.multiply %153, %101 : tensor<f32>
        %155 = stablehlo.multiply %154, %92 : tensor<f32>
        %156 = stablehlo.add %119, %137 : tensor<f32>
        %157 = stablehlo.add %156, %155 : tensor<f32>
        %158 = stablehlo.broadcast_in_dim %157, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
        %159 = stablehlo.convert %iterArg_24 : (tensor<i64>) -> tensor<i32>
        %c_52 = stablehlo.constant dense<1> : tensor<i32>
        %160 = stablehlo.convert %c_52 : tensor<i32>
        %161 = stablehlo.subtract %159, %160 : tensor<i32>
        %162 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32>
        %c_53 = stablehlo.constant dense<1> : tensor<i32>
        %163 = stablehlo.convert %c_53 : tensor<i32>
        %164 = stablehlo.subtract %162, %163 : tensor<i32>
        %165 = stablehlo.dynamic_update_slice %iterArg_22, %158, %161, %164 : (tensor<1024x1024xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1024x1024xf32>
        stablehlo.return %23, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %165, %iterArg_23, %iterArg_24, %iterArg_25 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x1024xf32>, tensor<1024x1024xf32>, tensor<f32>, tensor<i64>, tensor<1024xf32>
      }
      stablehlo.return %15, %iterArg_5, %iterArg_6, %iterArg_7, %19#4, %19#1, %19#5, %19#6, %19#8 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x1024xf32>, tensor<i64>, tensor<1024x1024xf32>, tensor<f32>, tensor<1024xf32>
    }
    %9 = stablehlo.transpose %8#6, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %10 = stablehlo.transpose %8#4, dims = [1, 0] : (tensor<3x1024xf32>) -> tensor<1024x3xf32>
    %11 = stablehlo.transpose %8#8, dims = [0] : (tensor<1024xf32>) -> tensor<1024xf32>
    return %9, %10, %11 : tensor<1024x1024xf32>, tensor<1024x3xf32>, tensor<1024xf32>
  }
}
