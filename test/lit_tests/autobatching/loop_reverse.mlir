// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x10x3xf32>) -> tensor<4x10x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x10x3xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<10> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<4x10x3xf32>) -> tensor<3x10x4xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<4x10x3xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_0 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %c, %4, %c, sizes = [3, 1, 4] : (tensor<3x10x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x4xf32>
      %6 = stablehlo.reshape %5 : (tensor<3x1x4xf32>) -> tensor<3x4xf32>
      %7 = stablehlo.reverse %6, dims = [0, 1] : tensor<3x4xf32>
      %8 = stablehlo.broadcast_in_dim %7, dims = [2, 0] : (tensor<3x4xf32>) -> tensor<4x1x3xf32>
      %9 = stablehlo.dynamic_update_slice %iterArg_4, %8, %c, %4, %c : (tensor<4x10x3xf32>, tensor<4x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x10x3xf32>
      stablehlo.return %2, %9 : tensor<i64>, tensor<4x10x3xf32>
    }
    return %1#1 : tensor<4x10x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x10x3xf32>) -> tensor<4x10x3xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<4x10x3xf32>) -> tensor<10x3x4xf32>
// CHECK-NEXT:     %1 = stablehlo.reverse %0, dims = [1, 2] : tensor<10x3x4xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<10x3x4xf32>) -> tensor<4x10x3xf32>
// CHECK-NEXT:     return %2 : tensor<4x10x3xf32>
// CHECK-NEXT: }
