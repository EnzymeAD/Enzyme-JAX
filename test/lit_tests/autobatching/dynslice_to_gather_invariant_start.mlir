// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

// out[i] = a[idx[i], 2]: the loop-invariant start index of the second
// dimension must survive the raising to a gather.
module {
  func.func @main(%arg0: tensor<5x2xf64>, %arg1: tensor<8xi64>) -> tensor<8xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<8xf64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c0, %iterArg_1 = %arg0, %iterArg_2 = %cst) : tensor<i64>, tensor<5x2xf64>, tensor<8xf64>
    cond {
      %1 = stablehlo.compare LT, %iterArg, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
      %2 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.dynamic_slice %arg1, %2, sizes = [1] : (tensor<8xi64>, tensor<i32>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
      %c_1 = stablehlo.constant dense<1> : tensor<i32>
      %6 = stablehlo.subtract %5, %c_1 : tensor<i32>
      %7 = stablehlo.dynamic_slice %iterArg_1, %6, %c_1, sizes = [1, 1] : (tensor<5x2xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
      %8 = stablehlo.reshape %7 : (tensor<1x1xf64>) -> tensor<1xf64>
      %9 = stablehlo.dynamic_update_slice %iterArg_2, %8, %2 : (tensor<8xf64>, tensor<1xf64>, tensor<i32>) -> tensor<8xf64>
      stablehlo.return %1, %iterArg_1, %9 : tensor<i64>, tensor<5x2xf64>, tensor<8xf64>
    }
    return %0#2 : tensor<8xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<5x2xf64>, %arg1: tensor<8xi64>) -> tensor<8xf64> {
// CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %[[C1V:.+]] = stablehlo.constant dense<1> : tensor<8xi32>
// CHECK-NEXT:    %[[CONV:.+]] = stablehlo.convert %arg1 : (tensor<8xi64>) -> tensor<8xi32>
// CHECK-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[CONV]], %[[C1V]] : tensor<8xi32>
// CHECK-NEXT:    %[[IDX:.+]] = stablehlo.reshape %[[SUB]] : (tensor<8xi32>) -> tensor<8x1xi32>
// CHECK-NEXT:    %[[IDX2:.+]] = stablehlo.pad %[[IDX]], %[[C1]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<8x1xi32>, tensor<i32>) -> tensor<8x2xi32>
// CHECK-NEXT:    %[[G:.+]] = "stablehlo.gather"(%arg0, %[[IDX2]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<5x2xf64>, tensor<8x2xi32>) -> tensor<8x1xf64>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.reshape %[[G]] : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<8xf64>
// CHECK-NEXT:  }
