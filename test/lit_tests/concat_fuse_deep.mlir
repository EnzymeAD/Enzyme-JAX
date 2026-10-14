// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_fuse" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A nest of concatenates along the same dim, several levels deep, flattens
// into one.
func.func @deep(%a: tensor<2x1xf32>, %b: tensor<2x2xf32>, %c: tensor<2x3xf32>, %d: tensor<2x4xf32>) -> tensor<2x10xf32> {
  %0 = stablehlo.concatenate %a, %b, dim = 1 : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  %1 = stablehlo.concatenate %0, %c, dim = 1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
  %2 = stablehlo.concatenate %1, %d, dim = 1 : (tensor<2x6xf32>, tensor<2x4xf32>) -> tensor<2x10xf32>
  return %2 : tensor<2x10xf32>
}

// CHECK:  func.func @deep(%arg0: tensor<2x1xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x3xf32>, %arg3: tensor<2x4xf32>) -> tensor<2x10xf32> {
// CHECK-NEXT:   %0 = stablehlo.concatenate %arg0, %arg1, %arg2, %arg3, dim = 1 : (tensor<2x1xf32>, tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x10xf32>
// CHECK-NEXT:   return %0 : tensor<2x10xf32>
// CHECK-NEXT: }

// A nest along another dim stays; an operand empty along dim goes.
func.func @other_dim(%a: tensor<1x2xf32>, %b: tensor<1x2xf32>, %c: tensor<2x3xf32>, %e: tensor<2x0xf32>) -> tensor<2x5xf32> {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.concatenate %e, %c, dim = 1 : (tensor<2x0xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
  return %2 : tensor<2x5xf32>
}

// CHECK:  func.func @other_dim(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<2x3xf32>, %arg3: tensor<2x0xf32>) -> tensor<2x5xf32> {
// CHECK-NEXT:   %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.concatenate %0, %arg2, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
// CHECK-NEXT:   return %1 : tensor<2x5xf32>
// CHECK-NEXT: }
