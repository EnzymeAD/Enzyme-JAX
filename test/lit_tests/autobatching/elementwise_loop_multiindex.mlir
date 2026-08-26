// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<22x20x12xf32>) -> tensor<22x20x12xf32> {
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<-2> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<22x20x12xf32>
    %c_2 = stablehlo.constant dense<20> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %c_4 = stablehlo.constant dense<2> : tensor<i64>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %c_6 = stablehlo.constant dense<3> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<22x20x12xf32>) -> tensor<12x20x22xf32>
    %1:2 = stablehlo.while(%iterArg = %c_5, %iterArg_8 = %cst) : tensor<i64>, tensor<22x20x12xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.multiply %iterArg, %c_6 : tensor<i64>
      %3 = stablehlo.add %c_7, %2 : tensor<i64>
      %4 = stablehlo.add %iterArg, %c_7 : tensor<i64>
      %5 = stablehlo.multiply %c_4, %3 : tensor<i64>
      %6 = stablehlo.add %5, %c_7 : tensor<i64>
      %7 = stablehlo.subtract %5, %c_7 : tensor<i64>
      %8 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c_3 : tensor<i32>
      %10 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
      %11 = stablehlo.subtract %10, %c_3 : tensor<i32>
      %12 = stablehlo.dynamic_slice %0, %c_0, %9, %11, sizes = [5, 1, 1] : (tensor<12x20x22xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x1xf32>
      %13 = stablehlo.add %3, %c_6 : tensor<i64>
      %14 = stablehlo.multiply %c_1, %3 : tensor<i64>
      %15 = stablehlo.add %c_2, %14 : tensor<i64>
      %16 = stablehlo.convert %13 : (tensor<i64>) -> tensor<i32>
      %17 = stablehlo.subtract %16, %c_3 : tensor<i32>
      %18 = stablehlo.convert %15 : (tensor<i64>) -> tensor<i32>
      %19 = stablehlo.subtract %18, %c_3 : tensor<i32>
      %20 = stablehlo.cosine %12 : tensor<5x1x1xf32>
      %21 = stablehlo.dynamic_update_slice %iterArg_8, %20, %c, %19, %17 : (tensor<22x20x12xf32>, tensor<5x1x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<22x20x12xf32>
      stablehlo.return %4, %21 : tensor<i64>, tensor<22x20x12xf32>
    }
    return %1#1 : tensor<22x20x12xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<22x20x12xf32>) -> tensor<22x20x12xf32> {
// CHECK-NEXT{LITERAL}:   %c = stablehlo.constant dense<[[2, 17, 3], [2, 11, 6], [2, 5, 9]]> : tensor<3x3xi32>
// CHECK-NEXT{LITERAL}:   %c_0 = stablehlo.constant dense<[[0, 2, 0], [0, 8, 6], [0, 14, 12]]> : tensor<3x3xi32>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<22x20x12xf32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<22x20x12xf32>) -> tensor<12x20x22xf32>
// CHECK-NEXT:   %1 = "stablehlo.gather"(%0, %c_0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 5, 1, 1>}> : (tensor<12x20x22xf32>, tensor<3x3xi32>) -> tensor<5x3xf32>
// CHECK-NEXT:   %2 = stablehlo.cosine %1 : tensor<5x3xf32>
// CHECK-NEXT:   %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<5x3xf32>) -> tensor<3x5xf32>
// CHECK-NEXT:   %4 = "stablehlo.scatter"(%cst, %c, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:     stablehlo.return %arg2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<22x20x12xf32>, tensor<3x3xi32>, tensor<3x5xf32>) -> tensor<22x20x12xf32>
// CHECK-NEXT:   return %4 : tensor<22x20x12xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %c = stablehlo.constant dense<10> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %cst) : tensor<i64>, tensor<10x10xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_0 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %4, %4, sizes = [1, 1] : (tensor<10x10xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %6 = stablehlo.multiply %5, %5 : tensor<1x1xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_3, %6, %4, %4 : (tensor<10x10xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<10x10xf32>
      stablehlo.return %2, %7 : tensor<i64>, tensor<10x10xf32>
    }
    return %1#1 : tensor<10x10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT{LITERAL}:   %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]> : tensor<10x2xi32>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:   %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<10x10xf32>, tensor<10x2xi32>) -> tensor<10xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %1 : tensor<10xf32>
// CHECK-NEXT:   %3 = "stablehlo.scatter"(%cst, %c, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:     stablehlo.return %arg2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<10x10xf32>, tensor<10x2xi32>, tensor<10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:   return %3 : tensor<10x10xf32>
// CHECK-NEXT: }
