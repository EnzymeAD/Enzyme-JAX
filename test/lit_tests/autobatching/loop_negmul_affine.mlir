// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x20x6xf32> {enzymexla.memory_effects = []}) -> tensor<5x20x6xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<-2> : tensor<i64>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x20x6xf32>
    %c_2 = stablehlo.constant dense<20> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %c_5 = stablehlo.constant dense<2> : tensor<i64>
    %c_6 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:2, 2:11:2, 0:4] : (tensor<5x20x6xf32>) -> tensor<2x5x4xf32>
    %1 = stablehlo.cosine %0 : tensor<2x5x4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 0, 3] : (tensor<2x5x4xf32>) -> tensor<5x2x1x4xf32>
    %3:2 = stablehlo.while(%iterArg = %c_4, %iterArg_7 = %cst) : tensor<i64>, tensor<5x20x6xf32> attributes {enzyme.disable_mincut}
    cond {
      %4 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    } do {
      %4 = stablehlo.multiply %iterArg, %c_5 : tensor<i64>
      %5 = stablehlo.add %c_6, %4 : tensor<i64>
      %6 = stablehlo.add %iterArg, %c_6 : tensor<i64>
      %7 = stablehlo.multiply %c, %5 : tensor<i64>
      %8 = stablehlo.add %c_2, %7 : tensor<i64>
      %9 = stablehlo.convert %8 : (tensor<i64>) -> tensor<i32>
      %10 = stablehlo.subtract %9, %c_3 : tensor<i32>
      %11 = stablehlo.dynamic_slice %2, %iterArg, %c_4, %c_4, %c_4, sizes = [1, 2, 1, 4] : (tensor<5x2x1x4xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x1x4xf32>
      %12 = stablehlo.reshape %11 : (tensor<1x2x1x4xf32>) -> tensor<2x1x4xf32>
      %13 = stablehlo.dynamic_update_slice %iterArg_7, %12, %c_0, %10, %c_3 : (tensor<5x20x6xf32>, tensor<2x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x20x6xf32>
      stablehlo.return %6, %13 : tensor<i64>, tensor<5x20x6xf32>
    }
    return %3#1 : tensor<5x20x6xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x20x6xf32> {enzymexla.memory_effects = []}) -> tensor<5x20x6xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[2, 17, 1], [2, 13, 1], [2, 9, 1], [2, 5, 1], [2, 1, 1]]> : tensor<5x3xi32>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x20x6xf32>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:2, 2:11:2, 0:4] : (tensor<5x20x6xf32>) -> tensor<2x5x4xf32>
// CHECK-NEXT:     %1 = stablehlo.cosine %0 : tensor<2x5x4xf32>
// CHECK-NEXT:     %2 = "stablehlo.scatter"(%cst, %c, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg2 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<5x20x6xf32>, tensor<5x3xi32>, tensor<2x5x4xf32>) -> tensor<5x20x6xf32>
// CHECK-NEXT:     return %2 : tensor<5x20x6xf32>
// CHECK-NEXT: }
