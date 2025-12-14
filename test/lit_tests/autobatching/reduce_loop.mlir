// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1);reshape_licm(1);transpose_dynamic_slice;transpose_licm(1);while_is_copy_simplify;reshape_elementwise(1);elementwise_licm(1);greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<7x5x2x3xf32>) -> tensor<5x4x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg1, dims = [3, 2, 1, 0] : (tensor<7x5x2x3xf32>) -> tensor<3x2x5x7xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<5x4x3xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_0 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %c, %4, %c, %c, sizes = [3, 1, 5, 7] : (tensor<3x2x5x7xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x5x7xf32>
      %6 = stablehlo.multiply %5, %5 : tensor<3x1x5x7xf32>
      %7 = stablehlo.reshape %6 : (tensor<3x1x5x7xf32>) -> tensor<3x5x7xf32>
      %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<3x5x7xf32>, tensor<f32>) -> tensor<3x5xf32>
      %9 = stablehlo.broadcast_in_dim %8, dims = [2, 0] : (tensor<3x5xf32>) -> tensor<5x1x3xf32>
      %10 = stablehlo.dynamic_update_slice %iterArg_4, %9, %c, %4, %c : (tensor<5x4x3xf32>, tensor<5x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x4x3xf32>
      stablehlo.return %2, %10 : tensor<i64>, tensor<5x4x3xf32>
    }
    return %1#1 : tensor<5x4x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<7x5x2x3xf32>) -> tensor<5x4x3xf32> {
// CHECK-NEXT:       %0 = stablehlo.dot_general %arg1, %arg1, batching_dims = [2, 3, 1] x [2, 3, 1], contracting_dims = [0] x [0] : (tensor<7x5x2x3xf32>, tensor<7x5x2x3xf32>) -> tensor<2x3x5xf32>
// CHECK-NEXT:       %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<2x3x5xf32>) -> tensor<5x2x3xf32>
// CHECK-NEXT:       %2 = stablehlo.slice %arg0 [0:5, 2:4, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x2x3xf32>
// CHECK-NEXT:       %3 = stablehlo.concatenate %1, %2, dim = 1 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x4x3xf32>
// CHECK-NEXT:       return %3 : tensor<5x4x3xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<7x5x2x3xf32>) -> tensor<5x4x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3, 2, 0, 1] : (tensor<7x5x2x3xf32>) -> tensor<2x3x5x7xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x3x5x7xf32>
    %2:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<5x4x3xf32>
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %1, %iterArg, %c_1, %c_1, %c_1, sizes = [1, 3, 5, 7] : (tensor<2x3x5x7xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x5x7xf32>
      %7 = stablehlo.reshape %6 : (tensor<1x3x5x7xf32>) -> tensor<3x5x7xf32>
      %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<3x5x7xf32>, tensor<f32>) -> tensor<3x5xf32>
      %9 = stablehlo.broadcast_in_dim %8, dims = [2, 0] : (tensor<3x5xf32>) -> tensor<5x1x3xf32>
      %10 = stablehlo.dynamic_update_slice %iterArg_4, %9, %c, %5, %c : (tensor<5x4x3xf32>, tensor<5x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x4x3xf32>
      stablehlo.return %3, %10 : tensor<i64>, tensor<5x4x3xf32>
    }
    return %2#1 : tensor<5x4x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<7x5x2x3xf32>) -> tensor<5x4x3xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg1, batching_dims = [2, 3, 1] x [2, 3, 1], contracting_dims = [0] x [0] : (tensor<7x5x2x3xf32>, tensor<7x5x2x3xf32>) -> tensor<2x3x5xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<2x3x5xf32>) -> tensor<5x2x3xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %arg0 [0:5, 2:4, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x2x3xf32>
// CHECK-NEXT:     %3 = stablehlo.concatenate %1, %2, dim = 1 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x4x3xf32>
// CHECK-NEXT:     return %3 : tensor<5x4x3xf32>
// CHECK-NEXT: }
