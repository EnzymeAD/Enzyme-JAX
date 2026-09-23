// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefix=NAN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=no_nan_zeros_scatter_multiply_simplify(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=TD-NONAN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=no_nan_zeros_scatter_multiply_simplify(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=TD-NAN

func.func @main(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
    %c = stablehlo.constant dense<1> : tensor<24x2xi64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %1 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
    %3 = stablehlo.reshape %1 : (tensor<24xi64>) -> tensor<24x1xi64>
    %4 = stablehlo.reshape %2 : (tensor<6x4xi64>) -> tensor<24x1xi64>
    %5 = stablehlo.concatenate %3, %4, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
    %6 = stablehlo.subtract %5, %c : tensor<24x2xi64>
    %7 = "stablehlo.scatter"(%cst_0, %6, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    %8 = stablehlo.multiply %7, %0 : tensor<1024x1024xf32>
    %9 = stablehlo.transpose %8, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %9 : tensor<1024x1024xf32>
}

// NONAN: func.func @main(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// NONAN:   %[[CST:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
// NONAN:   %[[CST_0:.*]] = stablehlo.constant dense<1> : tensor<24x2xi64>
// NONAN:   %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// NONAN:   %[[arg2_T:.*]] = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// NONAN:   %[[GATHER:.*]] = "stablehlo.gather"(%[[arg2_T]], %[[SCATTER_INDICES:.*]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<1024x1024xf32>, tensor<24x2xi64>) -> tensor<24xf32>
// NONAN:   %[[MUL:.*]] = stablehlo.multiply %[[CST]], %[[GATHER]] : tensor<24xf32>
// NONAN:   %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CST_1]], %[[SCATTER_INDICES]], %[[MUL]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// NONAN:   %[[RESULT:.*]] = stablehlo.transpose %[[SCATTER]], dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// NONAN:   return %[[RESULT]] : tensor<1024x1024xf32>
// NONAN: }

// The positions the scatter does not write are `0 * %arg2` in the result, which
// is only zero if %arg2 is neither NaN nor Inf, so without no_nan the multiply
// is left alone.
// NAN: func.func @main(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// NAN:   %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// NAN:   %[[arg2_T:.*]] = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// NAN:   %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CST_1]], %[[SCATTER_INDICES:.*]], %[[UPDATES:.*]])
// NAN:   %[[MUL:.*]] = stablehlo.multiply %[[SCATTER]], %[[arg2_T]]
// NAN:   %[[RESULT:.*]] = stablehlo.transpose %[[MUL]], dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// NAN:   return %[[RESULT]] : tensor<1024x1024xf32>
// NAN: }

// TD-NONAN: %[[GATHER:[0-9]+]] = "stablehlo.gather"(%[[arg2_T:[0-9]+]], %[[SCATTER_INDICES:[0-9]+]])
// TD-NONAN: %[[MUL:[0-9]+]] = stablehlo.multiply %[[CST:[a-z_0-9]+]], %[[GATHER]] : tensor<24xf32>
// TD-NONAN: %[[SCATTER:[0-9]+]] = "stablehlo.scatter"(%[[CST_0:[a-z_0-9]+]], %[[SCATTER_INDICES]], %[[MUL]])

// TD-NAN: %[[SCATTER:.*]] = "stablehlo.scatter"
// TD-NAN: stablehlo.multiply %[[SCATTER]],
