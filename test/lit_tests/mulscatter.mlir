// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-opt{passses=65536},enzyme-hlo-opt)' | FileCheck %s

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
    %7 = "stablehlo.scatter"(%cst_0, %6, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    %8 = stablehlo.multiply %7, %0 : tensor<1024x1024xf32>
    %9 = stablehlo.transpose %8, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %9 : tensor<1024x1024xf32>
}

// CHECK: func.func @main(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK:   %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// CHECK:   %[[CST:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
// CHECK:   %[[CST_0:.*]] = stablehlo.constant dense<1> : tensor<24x2xi64>
// CHECK:   %[[GATHER:.*]] = "stablehlo.gather"(%arg2, %[[SCATTER_INDICES:.*]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<1024x1024xf32>, tensor<24x2xi64>) -> tensor<24xf32>
// CHECK:   %[[MUL:.*]] = stablehlo.multiply %[[GATHER]], %[[CST]] : tensor<24xf32>
// CHECK:   %[[SCATTER:.*]] = "stablehlo.scatter"(%[[CST_1]], %[[SCATTER_INDICES]], %[[MUL]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK:   return %[[SCATTER]] : tensor<1024x1024xf32>
// CHECK: }
