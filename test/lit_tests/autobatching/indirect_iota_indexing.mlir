// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" --enzyme-hlo-generate-td="patterns=reshape_elementwise(1)" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
    %c_3 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst) : tensor<i64>, tensor<10xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>
      %2 = stablehlo.dynamic_slice %c_3, %iterArg, sizes = [1] : (tensor<10xi32>, tensor<i64>) -> tensor<1xi32>
      %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %5 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %6 = stablehlo.add %4, %5 : tensor<1xf64>
      %7 = stablehlo.maximum %4, %5 : tensor<1xf64>
      %8 = stablehlo.add %6, %7 : tensor<1xf64>
      %9 = stablehlo.convert %8 : (tensor<1xf64>) -> tensor<1xf32>
      %10 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %11 = stablehlo.subtract %10, %c : tensor<i32>
      %12 = stablehlo.dynamic_update_slice %iterArg_4, %9, %11 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %12 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf32> {
// CHECK-NEXT:     %0 = stablehlo.maximum %arg0, %arg1 : tensor<10xf64>
// CHECK-NEXT:     %1 = stablehlo.add %arg0, %arg1 : tensor<10xf64>
// CHECK-NEXT:     %2 = stablehlo.add %1, %0 : tensor<10xf64>
// CHECK-NEXT:     %3 = stablehlo.convert %2 : (tensor<10xf64>) -> tensor<10xf32>
// CHECK-NEXT:     return %3 : tensor<10xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<9x72x6xf32> {enzymexla.memory_effects = []}) -> tensor<9x72x6xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %c_1 = stablehlo.constant dense<7> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<9x72x6xf32>
    %c_2 = stablehlo.constant dense<[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]> : tensor<33xi64>
    %c_3 = stablehlo.constant dense<2> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i32>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %c_6 = stablehlo.constant dense<5> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<9x72x6xf32>) -> tensor<6x72x9xf32>
    %1:2 = stablehlo.while(%iterArg = %c_5, %iterArg_8 = %cst) : tensor<i64>, tensor<9x72x6xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.multiply %iterArg, %c_6 : tensor<i64>
      %3 = stablehlo.add %c_7, %2 : tensor<i64>
      %4 = stablehlo.add %iterArg, %c_7 : tensor<i64>
      %5 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %6 = stablehlo.subtract %5, %c_4 : tensor<i32>
      %7 = stablehlo.dynamic_slice %c_2, %6, sizes = [1] : (tensor<33xi64>, tensor<i32>) -> tensor<1xi64>
      %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
      %9 = stablehlo.dynamic_slice %0, %c_4, %6, %c_0, sizes = [3, 1, 4] : (tensor<6x72x9xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x4xf32>
      %10 = stablehlo.sine %9 : tensor<3x1x4xf32>
      %11 = stablehlo.multiply %c_3, %8 : tensor<i64>
      %12 = stablehlo.add %11, %c_7 : tensor<i64>
      %13 = stablehlo.convert %12 : (tensor<i64>) -> tensor<i32>
      %14 = stablehlo.subtract %13, %c_4 : tensor<i32>
      %15 = stablehlo.transpose %10, dims = [2, 1, 0] : (tensor<3x1x4xf32>) -> tensor<4x1x3xf32>
      %16 = stablehlo.dynamic_update_slice %iterArg_8, %15, %c, %14, %c_0 : (tensor<9x72x6xf32>, tensor<4x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x72x6xf32>
      stablehlo.return %4, %16 : tensor<i64>, tensor<9x72x6xf32>
    }
    return %1#1 : tensor<9x72x6xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<9x72x6xf32> {enzymexla.memory_effects = []}) -> tensor<9x72x6xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[0, 10, 2], [0, 20, 2], [0, 30, 2], [0, 40, 2], [0, 50, 2], [0, 60, 2], [0, 70, 2]]> : tensor<7x3xi32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<9x72x6xf32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [2:6, 0:31:5, 1:4] : (tensor<9x72x6xf32>) -> tensor<4x7x3xf32>
// CHECK-NEXT:    %1 = stablehlo.sine %0 : tensor<4x7x3xf32>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<4x7x3xf32>) -> tensor<7x4x3xf32>
// CHECK-NEXT:    %3 = "stablehlo.scatter"(%cst, %c, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %arg2 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<9x72x6xf32>, tensor<7x3xi32>, tensor<7x4x3xf32>) -> tensor<9x72x6xf32>
// CHECK-NEXT:    return %3 : tensor<9x72x6xf32>
// CHECK-NEXT:  }
