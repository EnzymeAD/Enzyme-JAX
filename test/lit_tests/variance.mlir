// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.reshape_transpose_to_broadcast
    } : !transform.any_op
    transform.yield 
  }
  func.func @var(%arg0: tensor<4x3x2xf64> ) -> (tensor<4x1x1xf64>, tensor<4x3x2xf64>) {
    %cst = stablehlo.constant dense<0.16666666666666666> : tensor<4x1x1xf64>
    %cst_0 = stablehlo.constant dense<0.16666666666666666> : tensor<1x1x4xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<4x3x2xf64>) -> tensor<2x3x4xf64>
    %1 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.add across dimensions = [2, 1] : (tensor<4x3x2xf64>, tensor<f64>) -> tensor<4xf64>
    // CHECK: %2 = stablehlo.broadcast_in_dim %1, dims = [2] : (tensor<4xf64>) -> tensor<1x1x4xf64>
    // CHECK-NOT: stablehlo.transpose
    // CHECK-NOT: stablehlo.reshape
    // CHECK: %6 = stablehlo.dot_general %5, %5, batching_dims = [2] x [2], contracting_dims = [0, 1] x [0, 1] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<4xf64>
    %2 = stablehlo.transpose %1, dims = [0] : (tensor<4xf64>) -> tensor<4xf64>
    %3 = stablehlo.reshape %2 : (tensor<4xf64>) -> tensor<1x1x4xf64>
    %4 = stablehlo.multiply %3, %cst_0 : tensor<1x1x4xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<1x1x4xf64>) -> tensor<2x3x4xf64>
    %6 = stablehlo.subtract %0, %5 : tensor<2x3x4xf64>
    %7 = stablehlo.dot_general %6, %6, batching_dims = [2] x [2], contracting_dims = [0, 1] x [0, 1] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<4xf64>
    %8 = stablehlo.reshape %7 : (tensor<4xf64>) -> tensor<4x1x1xf64>
    %9 = stablehlo.multiply %8, %cst : tensor<4x1x1xf64>
    return %9, %arg0 : tensor<4x1x1xf64>, tensor<4x3x2xf64>
  }
}

