// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.dot_general_reshape
    } : !transform.any_op
    transform.yield
  }
  func.func @f_generator(%arg0: tensor<6x2xf32>, %arg1: tensor<2x4xf32>) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x2xf32>) -> tensor<2x6xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %2 = stablehlo.reshape %arg1 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %3 = stablehlo.reshape %arg1 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %4 = stablehlo.reshape %arg1 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %5 = stablehlo.reshape %arg1 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %6 = stablehlo.reshape %arg1 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %7 = stablehlo.concatenate %1, %2, %3, %4, %5, %6, dim = 0 : (tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<6x2x4xf32>
    %8 = stablehlo.reshape %0 : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %9 = stablehlo.dot_general %7, %8, batching_dims = [0] x [1], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x2x4xf32>, tensor<2x6x1xf32>) -> tensor<6x4x1xf32>
    %10 = stablehlo.slice %9 [0:1, 0:4, 0:1] : (tensor<6x4x1xf32>) -> tensor<1x4x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<1x4x1xf32>) -> tensor<4x1xf32>
    %12 = stablehlo.slice %9 [1:2, 0:4, 0:1] : (tensor<6x4x1xf32>) -> tensor<1x4x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x4x1xf32>) -> tensor<4x1xf32>
    %14 = stablehlo.slice %9 [2:3, 0:4, 0:1] : (tensor<6x4x1xf32>) -> tensor<1x4x1xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x4x1xf32>) -> tensor<4x1xf32>
    %16 = stablehlo.slice %9 [3:4, 0:4, 0:1] : (tensor<6x4x1xf32>) -> tensor<1x4x1xf32>
    %17 = stablehlo.reshape %16 : (tensor<1x4x1xf32>) -> tensor<4x1xf32>
    %18 = stablehlo.slice %9 [4:5, 0:4, 0:1] : (tensor<6x4x1xf32>) -> tensor<1x4x1xf32>
    %19 = stablehlo.reshape %18 : (tensor<1x4x1xf32>) -> tensor<4x1xf32>
    %20 = stablehlo.slice %9 [5:6, 0:4, 0:1] : (tensor<6x4x1xf32>) -> tensor<1x4x1xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x4x1xf32>) -> tensor<4x1xf32>
    %22 = stablehlo.reshape %11 : (tensor<4x1xf32>) -> tensor<4xf32>
    %23 = stablehlo.reshape %13 : (tensor<4x1xf32>) -> tensor<4xf32>
    %24 = stablehlo.reshape %15 : (tensor<4x1xf32>) -> tensor<4xf32>
    %25 = stablehlo.reshape %17 : (tensor<4x1xf32>) -> tensor<4xf32>
    %26 = stablehlo.reshape %19 : (tensor<4x1xf32>) -> tensor<4xf32>
    %27 = stablehlo.reshape %21 : (tensor<4x1xf32>) -> tensor<4xf32>
    %28 = stablehlo.add %22, %23 : tensor<4xf32>
    %29 = stablehlo.add %28, %24 : tensor<4xf32>
    %30 = stablehlo.add %29, %25 : tensor<4xf32>
    %31 = stablehlo.add %30, %26 : tensor<4xf32>
    %32 = stablehlo.add %31, %27 : tensor<4xf32>
    return %32, %arg0, %arg1 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
  }
}

// CHECK: func.func @f_generator(%arg0: tensor<6x2xf32>, %arg1: tensor<2x4xf32>) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<2x4xf32>) -> tensor<6x2x4xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %0, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x2x4xf32>, tensor<6x2xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<6x4xf32>) -> tensor<6x4x1xf32>
// CHECK-NEXT:   %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<4x1xf32>) -> tensor<4xf32>
// CHECK-NEXT:   return %4, %arg0, %arg1 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
// CHECK-NEXT: }
