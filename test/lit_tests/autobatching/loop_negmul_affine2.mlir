// RUN: enzymexlamlir-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x20x6xf32>) -> (tensor<5x20x6xf32>) {
    %c = stablehlo.constant dense<-2> : tensor<i64>
    %c_0 = stablehlo.constant dense<-3> : tensor<i64>
    %c_1 = stablehlo.constant dense<2> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %c_4 = stablehlo.constant dense<20> : tensor<i64>
    %c_5 = stablehlo.constant dense<3> : tensor<i64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_7 = stablehlo.constant dense<2> : tensor<i64>
    %c_8 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x20x5xf32>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x20x6xf32>) -> tensor<6x20x5xf32>
    %1:2 = stablehlo.while(%iterArg = %c_6, %iterArg_9 = %cst) : tensor<i64>, tensor<6x20x5xf32>
    cond {
      %4 = stablehlo.compare  LT, %iterArg, %c_5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    } do {
      %4 = stablehlo.multiply %iterArg, %c_7 : tensor<i64>
      %5 = stablehlo.add %c_8, %4 : tensor<i64>
      %6 = stablehlo.add %iterArg, %c_8 : tensor<i64>
      %7 = stablehlo.multiply %c_0, %5 : tensor<i64>
      %8 = stablehlo.add %c_4, %7 : tensor<i64>
      %9 = stablehlo.convert %8 : (tensor<i64>) -> tensor<i32>
      %10 = stablehlo.subtract %9, %c_2 : tensor<i32>
      %11 = stablehlo.dynamic_slice %0, %c_3, %10, %c_3, sizes = [4, 1, 2] : (tensor<6x20x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x2xf32>
      %12 = stablehlo.multiply %c, %5 : tensor<i64>
      %13 = stablehlo.add %c_4, %12 : tensor<i64>
      %14 = stablehlo.cosine %11 : tensor<4x1x2xf32>
      %15 = stablehlo.convert %13 : (tensor<i64>) -> tensor<i32>
      %16 = stablehlo.subtract %15, %c_2 : tensor<i32>
      %17 = stablehlo.dynamic_update_slice %iterArg_9, %14, %c_2, %16, %c_1 : (tensor<6x20x5xf32>, tensor<4x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x20x5xf32>
      stablehlo.return %6, %17 : tensor<i64>, tensor<6x20x5xf32>
    }
    %2 = stablehlo.transpose %1#1, dims = [2, 1, 0] : (tensor<6x20x5xf32>) -> tensor<5x20x6xf32>
    return %2 : tensor<5x20x6xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x20x6xf32>) -> tensor<5x20x6xf32> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[1, 17, 2], [1, 13, 2], [1, 9, 2]]> : tensor<3x3xi32>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x20x5xf32>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:2, 4:17:6, 0:4] : (tensor<5x20x6xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
// CHECK-NEXT:     %2 = stablehlo.reverse %1, dims = [1] : tensor<4x3x2xf32>
// CHECK-NEXT:     %3 = stablehlo.broadcast_in_dim %2, dims = [1, 0, 3] : (tensor<4x3x2xf32>) -> tensor<3x4x1x2xf32>
// CHECK-NEXT:     %4 = stablehlo.cosine %3 : tensor<3x4x1x2xf32>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<3x4x1x2xf32>) -> tensor<3x4x2xf32>
// CHECK-NEXT:     %6 = stablehlo.transpose %5, dims = [1, 0, 2] : (tensor<3x4x2xf32>) -> tensor<4x3x2xf32>
// CHECK-NEXT:     %7 = "stablehlo.scatter"(%cst, %c, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg2 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<6x20x5xf32>, tensor<3x3xi32>, tensor<4x3x2xf32>) -> tensor<6x20x5xf32>
// CHECK-NEXT:     %8 = stablehlo.transpose %7, dims = [2, 1, 0] : (tensor<6x20x5xf32>) -> tensor<5x20x6xf32>
// CHECK-NEXT:     return %8 : tensor<5x20x6xf32>
// CHECK-NEXT: }
