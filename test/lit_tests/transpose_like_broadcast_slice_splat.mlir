// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_broadcast;transpose_like_broadcast_slice" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A transpose-like broadcast of a sliced splat constant: pushed above the
// slice it would broadcast the splat itself, which SliceBroadcast pushes back
// below the slice. Neither direction may be taken.
func.func @sliced_splat(%y: tensor<1x4x3xf64>) -> tensor<1x4x3xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
  %0 = stablehlo.slice %cst [0:3, 0:4] : (tensor<4x4xf64>) -> tensor<3x4xf64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1] : (tensor<3x4xf64>) -> tensor<1x4x3xf64>
  %2 = stablehlo.add %1, %y : tensor<1x4x3xf64>
  return %2 : tensor<1x4x3xf64>
}

// CHECK:  func.func @sliced_splat(%arg0: tensor<1x4x3xf64>) -> tensor<1x4x3xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
// CHECK-NEXT:   %0 = stablehlo.slice %cst [0:3, 0:4] : (tensor<4x4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1] : (tensor<3x4xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:   %2 = stablehlo.add %1, %arg0 : tensor<1x4x3xf64>
// CHECK-NEXT:   return %2 : tensor<1x4x3xf64>
// CHECK-NEXT: }

// -----

// The same broadcast of a sliced argument is a transpose and moves up.
func.func @sliced_arg(%x: tensor<4x4xf64>, %y: tensor<1x4x3xf64>) -> tensor<1x4x3xf64> {
  %0 = stablehlo.slice %x [0:3, 0:4] : (tensor<4x4xf64>) -> tensor<3x4xf64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1] : (tensor<3x4xf64>) -> tensor<1x4x3xf64>
  %2 = stablehlo.add %1, %y : tensor<1x4x3xf64>
  return %2 : tensor<1x4x3xf64>
}

// CHECK:  func.func @sliced_arg(%arg0: tensor<4x4xf64>, %arg1: tensor<1x4x3xf64>) -> tensor<1x4x3xf64> {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 1] : (tensor<4x4xf64>) -> tensor<1x4x4xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:1, 0:4, 0:3] : (tensor<1x4x4xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:   %2 = stablehlo.add %1, %arg1 : tensor<1x4x3xf64>
// CHECK-NEXT:   return %2 : tensor<1x4x3xf64>
// CHECK-NEXT: }
