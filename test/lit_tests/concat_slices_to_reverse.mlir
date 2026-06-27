// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// 3-D tensor: slices along dim 0 in reverse order → stablehlo.reverse
// CHECK-LABEL: func.func @reverse_dim0
func.func @reverse_dim0(%arg0: tensor<5x3x10xf32>) -> tensor<5x3x10xf32> {
  %0 = stablehlo.slice %arg0 [4:5, 0:3, 0:10] : (tensor<5x3x10xf32>) -> tensor<1x3x10xf32>
  %1 = stablehlo.slice %arg0 [3:4, 0:3, 0:10] : (tensor<5x3x10xf32>) -> tensor<1x3x10xf32>
  %2 = stablehlo.slice %arg0 [2:3, 0:3, 0:10] : (tensor<5x3x10xf32>) -> tensor<1x3x10xf32>
  %3 = stablehlo.slice %arg0 [1:2, 0:3, 0:10] : (tensor<5x3x10xf32>) -> tensor<1x3x10xf32>
  %4 = stablehlo.slice %arg0 [0:1, 0:3, 0:10] : (tensor<5x3x10xf32>) -> tensor<1x3x10xf32>
  %5 = stablehlo.concatenate %0, %1, %2, %3, %4, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
  return %5 : tensor<5x3x10xf32>
  // CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [0]
  // CHECK-NEXT: return %[[R]]
}

// 1-D tensor: simple reverse
// CHECK-LABEL: func.func @reverse_1d
func.func @reverse_1d(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
  %1 = stablehlo.slice %arg0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
  %2 = stablehlo.slice %arg0 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
  %3 = stablehlo.slice %arg0 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
  %4 = stablehlo.concatenate %0, %1, %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
  return %4 : tensor<4xf32>
  // CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [0]
  // CHECK-NEXT: return %[[R]]
}

// Negative: slices NOT in reverse order → must not fold
// CHECK-LABEL: func.func @no_fold_wrong_order
func.func @no_fold_wrong_order(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.slice %arg0 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
  %1 = stablehlo.slice %arg0 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
  %2 = stablehlo.slice %arg0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
  %3 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
  %4 = stablehlo.concatenate %0, %1, %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
  return %4 : tensor<4xf32>
  // CHECK-NOT: stablehlo.reverse
}

// Negative: one operand is not a slice → must not fold
// CHECK-LABEL: func.func @no_fold_not_all_slices
func.func @no_fold_not_all_slices(%arg0: tensor<4xf32>, %arg1: tensor<1xf32>) -> tensor<4xf32> {
  %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
  %1 = stablehlo.slice %arg0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
  %2 = stablehlo.slice %arg0 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
  // fourth operand is a plain argument, not a slice
  %3 = stablehlo.concatenate %0, %1, %2, %arg1, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
  // CHECK-NOT: stablehlo.reverse
}

// Negative: slices come from different source tensors → must not fold
// CHECK-LABEL: func.func @no_fold_different_sources
func.func @no_fold_different_sources(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
  %1 = stablehlo.slice %arg0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
  %2 = stablehlo.slice %arg1 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
  %3 = stablehlo.slice %arg0 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
  %4 = stablehlo.concatenate %0, %1, %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
  return %4 : tensor<4xf32>
  // CHECK-NOT: stablehlo.reverse
}

// Negative: fewer slices than source extent (N=3, extent=4) → must not fold
// CHECK-LABEL: func.func @no_fold_incomplete_coverage
func.func @no_fold_incomplete_coverage(%arg0: tensor<4xf32>) -> tensor<3xf32> {
  %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
  %1 = stablehlo.slice %arg0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
  %2 = stablehlo.slice %arg0 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
  // [0:1] is missing; N=3 but source extent=4
  %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
  return %3 : tensor<3xf32>
  // CHECK-NOT: stablehlo.reverse
}
