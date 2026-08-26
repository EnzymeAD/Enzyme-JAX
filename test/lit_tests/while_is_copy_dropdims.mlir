// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<5x4x3xf32>) -> tensor<4x10x5xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x10x5xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [1] x [0], contracting_dims = [0] x [2] : (tensor<3x5x10xf32>, tensor<5x4x3xf32>) -> tensor<5x10x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2, 1] : (tensor<5x10x4xf32>) -> tensor<5x4x10x1xf32>
    %2:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst) : tensor<i64>, tensor<4x10x5xf32>
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_1 : tensor<i32>
      %6 = stablehlo.dynamic_slice %1, %iterArg, %c_2, %c_2, %c_2, sizes = [1, 4, 10, 1] : (tensor<5x4x10x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x4x10x1xf32>
      %7 = stablehlo.reshape %6 : (tensor<1x4x10x1xf32>) -> tensor<4x10x1xf32>
      %8 = stablehlo.dynamic_update_slice %iterArg_4, %7, %c, %c, %5 : (tensor<4x10x5xf32>, tensor<4x10x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x10x5xf32>
      stablehlo.return %3, %8 : tensor<i64>, tensor<4x10x5xf32>
    }
    return %2#1 : tensor<4x10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<5x4x3xf32>) -> tensor<4x10x5xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [1] x [0], contracting_dims = [0] x [2] : (tensor<3x5x10xf32>, tensor<5x4x3xf32>) -> tensor<5x10x4xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<5x10x4xf32>) -> tensor<4x10x5xf32>
// CHECK-NEXT:     return %1 : tensor<4x10x5xf32>
// CHECK-NEXT: }
