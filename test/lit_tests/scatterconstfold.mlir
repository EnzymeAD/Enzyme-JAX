// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=scatter_const_fold(1024)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=APPLY
// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=scatter_const_fold(1)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=NOAPPLY

module @reactant_unfold attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<1x1x7xf32> {enzymexla.memory_effects = []}) -> tensor<1x3x4xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2]]> : tensor<3x2xi64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x7xf32>) -> tensor<7x1x1xf32>
    %1 = "stablehlo.scatter"(%cst_1, %c, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      stablehlo.return %cst : tensor<f32>
    }) : (tensor<3x3xf32>, tensor<3x2xi64>, tensor<3xf32>) -> tensor<3x3xf32>
    %2 = stablehlo.reshape %1 : (tensor<3x3xf32>) -> tensor<3x1x3xf32>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [0, f, b]x[0, i, o]->[b, f, 0], window = {stride = [2], pad = [[1, 1]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<7x1x1xf32>, tensor<3x1x3xf32>) -> tensor<1x3x4xf32>
    return %3 : tensor<1x3x4xf32>
  }
}

// APPLY: func.func @main(%arg0: tensor<1x1x7xf32> {enzymexla.memory_effects = []}) -> tensor<1x3x4xf32> attributes {enzymexla.memory_effects = []} {
// APPLY-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf32>
// APPLY-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<1x1x7xf32>) -> tensor<7x1x1xf32>
// APPLY-NEXT:     %1 = stablehlo.reshape %cst : (tensor<3x3xf32>) -> tensor<3x1x3xf32>
// APPLY-NEXT{LITERAL}:     %2 = stablehlo.convolution(%0, %1) dim_numbers = [0, f, b]x[0, i, o]->[b, f, 0], window = {stride = [2], pad = [[1, 1]], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<7x1x1xf32>, tensor<3x1x3xf32>) -> tensor<1x3x4xf32>
// APPLY-NEXT:     return %2 : tensor<1x3x4xf32>
// APPLY-NEXT: }

// NOAPPLY: stablehlo.scatter
