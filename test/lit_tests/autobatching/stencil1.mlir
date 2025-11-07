// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x12x4xf32> {enzymexla.memory_effects = []}) -> tensor<5x8x4xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<8> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x8x4xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %c_5 = stablehlo.constant dense<3> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x12x4xf32>) -> tensor<4x12x5xf32>
    %1:2 = stablehlo.while(%iterArg = %c_3, %iterArg_6 = %cst) : tensor<i64>, tensor<5x8x4xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %3 = stablehlo.add %iterArg, %c_4 : tensor<i64>
      %4 = stablehlo.subtract %2, %c_2 : tensor<i64>
      %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
      %6 = stablehlo.subtract %5, %c_1 : tensor<i32>
      %7 = stablehlo.dynamic_slice %0, %c, %6, %c, sizes = [4, 1, 5] : (tensor<4x12x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x5xf32>
      %8 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c_1 : tensor<i32>
      %10 = stablehlo.dynamic_slice %0, %c, %9, %c, sizes = [4, 1, 5] : (tensor<4x12x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x5xf32>
      %11 = stablehlo.add %2, %c_2 : tensor<i64>
      %12 = stablehlo.convert %11 : (tensor<i64>) -> tensor<i32>
      %13 = stablehlo.subtract %12, %c_1 : tensor<i32>
      %14 = stablehlo.dynamic_slice %0, %c, %13, %c, sizes = [4, 1, 5] : (tensor<4x12x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x5xf32>
      %15 = stablehlo.add %7, %10 : tensor<4x1x5xf32>
      %16 = stablehlo.subtract %15, %14 : tensor<4x1x5xf32>
      %17 = stablehlo.transpose %16, dims = [2, 1, 0] : (tensor<4x1x5xf32>) -> tensor<5x1x4xf32>
      %18 = stablehlo.dynamic_update_slice %iterArg_6, %17, %c, %6, %c : (tensor<5x8x4xf32>, tensor<5x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x8x4xf32>
      stablehlo.return %3, %18 : tensor<i64>, tensor<5x8x4xf32>
    }
    return %1#1 : tensor<5x8x4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x12x4xf32> {enzymexla.memory_effects = []}) -> tensor<5x8x4xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:5, 0:8, 0:4] : (tensor<5x12x4xf32>) -> tensor<5x8x4xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %arg0 [0:5, 2:10, 0:4] : (tensor<5x12x4xf32>) -> tensor<5x8x4xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %arg0 [0:5, 4:12, 0:4] : (tensor<5x12x4xf32>) -> tensor<5x8x4xf32>
// CHECK-NEXT:     %3 = stablehlo.broadcast_in_dim %0, dims = [3, 0, 1] : (tensor<5x8x4xf32>) -> tensor<8x4x1x5xf32>
// CHECK-NEXT:     %4 = stablehlo.broadcast_in_dim %1, dims = [3, 0, 1] : (tensor<5x8x4xf32>) -> tensor<8x4x1x5xf32>
// CHECK-NEXT:     %5 = stablehlo.add %3, %4 : tensor<8x4x1x5xf32>
// CHECK-NEXT:     %6 = stablehlo.broadcast_in_dim %2, dims = [3, 0, 1] : (tensor<5x8x4xf32>) -> tensor<8x4x1x5xf32>
// CHECK-NEXT:     %7 = stablehlo.subtract %5, %6 : tensor<8x4x1x5xf32>
// CHECK-NEXT:     %8 = stablehlo.transpose %7, dims = [0, 3, 2, 1] : (tensor<8x4x1x5xf32>) -> tensor<8x5x1x4xf32>
// CHECK-NEXT:     %9 = stablehlo.reshape %8 : (tensor<8x5x1x4xf32>) -> tensor<8x5x4xf32>
// CHECK-NEXT:     %10 = stablehlo.transpose %9, dims = [1, 0, 2] : (tensor<8x5x4xf32>) -> tensor<5x8x4xf32>
// CHECK-NEXT:     return %10 : tensor<5x8x4xf32>
// CHECK-NEXT:   }
