// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x7x3xf32> {enzymexla.memory_effects = []}, %arg1: tensor<4x7x3xf32> {enzymexla.memory_effects = []}) -> tensor<8x7x3xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<8x7x3xf32>
    %c_3 = stablehlo.constant dense<7> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<8x7x3xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c_0 : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %c, %3, %c, sizes = [4, 1, 3] : (tensor<4x7x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x3xf32>
      %5 = stablehlo.dynamic_slice %arg1, %c, %3, %c, sizes = [4, 1, 3] : (tensor<4x7x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x3xf32>
      %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<4x1x3xf32>, tensor<4x1x3xf32>) -> tensor<8x1x3xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_4, %6, %c, %3, %c : (tensor<8x7x3xf32>, tensor<8x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x7x3xf32>
      stablehlo.return %1, %7 : tensor<i64>, tensor<8x7x3xf32>
    }
    return %0#1 : tensor<8x7x3xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x7x3xf32> {enzymexla.memory_effects = []}, %arg1: tensor<4x7x3xf32> {enzymexla.memory_effects = []}) -> tensor<8x7x3xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0, 3] : (tensor<4x7x3xf32>) -> tensor<7x4x1x3xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1, 0, 3] : (tensor<4x7x3xf32>) -> tensor<7x4x1x3xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<7x4x1x3xf32>, tensor<7x4x1x3xf32>) -> tensor<7x8x1x3xf32>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<7x8x1x3xf32>) -> tensor<7x8x3xf32>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<7x8x3xf32>) -> tensor<8x7x3xf32>
// CHECK-NEXT:    return %4 : tensor<8x7x3xf32>
// CHECK-NEXT:  }
