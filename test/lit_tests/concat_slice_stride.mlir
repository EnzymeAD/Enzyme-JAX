// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_slice<1>" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Unit stride: the second slice starts at the first's limit.
func.func @unit(%x: tensor<6xf64>) -> tensor<2xf64> {
  %s0 = stablehlo.slice %x [4:5] : (tensor<6xf64>) -> tensor<1xf64>
  %s1 = stablehlo.slice %x [5:6] : (tensor<6xf64>) -> tensor<1xf64>
  %cc = stablehlo.concatenate %s0, %s1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  return %cc : tensor<2xf64>
}

// CHECK:  func.func @unit(%arg0: tensor<6xf64>) -> tensor<2xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [4:6] : (tensor<6xf64>) -> tensor<2xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %0, dim = 0 : (tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:   return %1 : tensor<2xf64>
// CHECK-NEXT: }

// -----

// Stride 2: [0:5:2] takes 0, 2, 4 and [6:9:2] takes 6, 8, so together they are
// [0:9:2].
func.func @strided(%x: tensor<9xf64>) -> tensor<5xf64> {
  %s0 = stablehlo.slice %x [0:5:2] : (tensor<9xf64>) -> tensor<3xf64>
  %s1 = stablehlo.slice %x [6:9:2] : (tensor<9xf64>) -> tensor<2xf64>
  %cc = stablehlo.concatenate %s0, %s1, dim = 0 : (tensor<3xf64>, tensor<2xf64>) -> tensor<5xf64>
  return %cc : tensor<5xf64>
}

// CHECK:  func.func @strided(%arg0: tensor<9xf64>) -> tensor<5xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:9:2] : (tensor<9xf64>) -> tensor<5xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %0, dim = 0 : (tensor<5xf64>) -> tensor<5xf64>
// CHECK-NEXT:   return %1 : tensor<5xf64>
// CHECK-NEXT: }

// -----

// Stride 2 with the second slice starting at the first's limit: [4:5:2] takes
// 4 and [5:6:2] takes 5, which no single stride-2 slice yields.
func.func @limit_is_not_next(%x: tensor<6xf64>) -> tensor<2xf64> {
  %s0 = stablehlo.slice %x [4:5:2] : (tensor<6xf64>) -> tensor<1xf64>
  %s1 = stablehlo.slice %x [5:6:2] : (tensor<6xf64>) -> tensor<1xf64>
  %cc = stablehlo.concatenate %s0, %s1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  return %cc : tensor<2xf64>
}

// CHECK:  func.func @limit_is_not_next(%arg0: tensor<6xf64>) -> tensor<2xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [4:5:2] : (tensor<6xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [5:6:2] : (tensor<6xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
// CHECK-NEXT:   return %2 : tensor<2xf64>
// CHECK-NEXT: }
