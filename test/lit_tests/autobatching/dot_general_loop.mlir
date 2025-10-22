// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<3x5x10xf32> {enzymexla.memory_effects = []}, %arg1: tensor<4x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x5x10xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x5x10xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %1:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst) : tensor<i64>, tensor<4x5x10xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_1 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %c, %4, %c, sizes = [10, 1, 3] : (tensor<10x5x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x3xf32>
      %6 = stablehlo.reshape %5 : (tensor<10x1x3xf32>) -> tensor<10x3xf32>
      %7 = stablehlo.dot_general %6, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<10x3xf32>, tensor<4x3xf32>) -> tensor<10x4xf32>
      %8 = stablehlo.broadcast_in_dim %7, dims = [2, 0] : (tensor<10x4xf32>) -> tensor<4x1x10xf32>
      %9 = stablehlo.dynamic_update_slice %iterArg_4, %8, %c, %4, %c : (tensor<4x5x10xf32>, tensor<4x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x5x10xf32>
      stablehlo.return %2, %9 : tensor<i64>, tensor<4x5x10xf32>
    }
    return %1#1 : tensor<4x5x10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x5x10xf32> {enzymexla.memory_effects = []}, %arg1: tensor<4x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x5x10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<4x3xf32>) -> tensor<5x4x3xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg0, %0, batching_dims = [1] x [0], contracting_dims = [0] x [2], precision = [DEFAULT, DEFAULT] : (tensor<3x5x10xf32>, tensor<5x4x3xf32>) -> tensor<5x10x4xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<5x10x4xf32>) -> tensor<4x5x10xf32>
// CHECK-NEXT:   return %2 : tensor<4x5x10xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<3x5x10xf32> {enzymexla.memory_effects = []}, %arg1: tensor<5x4x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x10x5xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x10x5xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<5x4x3xf32>) -> tensor<3x4x5xf32>
    %2:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst) : tensor<i64>, tensor<4x10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_1 : tensor<i32>
      %6 = stablehlo.dynamic_slice %0, %c, %5, %c, sizes = [10, 1, 3] : (tensor<10x5x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x3xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x3xf32>) -> tensor<10x3xf32>
      %8 = stablehlo.dynamic_slice %1, %c, %c, %5, sizes = [3, 4, 1] : (tensor<3x4x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x1xf32>
      %9 = stablehlo.reshape %8 : (tensor<3x4x1xf32>) -> tensor<3x4xf32>
      %10 = stablehlo.dot_general %7, %9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x3xf32>, tensor<3x4xf32>) -> tensor<10x4xf32>
      %11 = stablehlo.broadcast_in_dim %10, dims = [1, 0] : (tensor<10x4xf32>) -> tensor<4x10x1xf32>
      %12 = stablehlo.dynamic_update_slice %iterArg_4, %11, %c, %c, %5 : (tensor<4x10x5xf32>, tensor<4x10x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x10x5xf32>
      stablehlo.return %3, %12 : tensor<i64>, tensor<4x10x5xf32>
    }
    return %2#1 : tensor<4x10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x5x10xf32> {enzymexla.memory_effects = []}, %arg1: tensor<5x4x3xf32> {enzymexla.memory_effects = []}) -> tensor<4x10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [1] x [0], contracting_dims = [0] x [2], precision = [DEFAULT, DEFAULT] : (tensor<3x5x10xf32>, tensor<5x4x3xf32>) -> tensor<5x10x4xf32>
// CHECK-NEXT:   %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<5x10x4xf32>) -> tensor<4x10x5xf32>
// CHECK-NEXT:   return %1 : tensor<4x10x5xf32>
// CHECK-NEXT: }
