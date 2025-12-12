// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @full_overlap(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<4x3xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
  %0 = stablehlo.dynamic_update_slice %cst, %arg0, %arg1, %arg2 : (tensor<32x32xf32>, tensor<4x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
  %1 = stablehlo.dynamic_slice %0, %arg1, %arg2, sizes = [4, 3] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}

// CHECK: func.func @full_overlap(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x3xf32> {
// CHECK-NEXT:   return %arg0 : tensor<4x3xf32>
// CHECK-NEXT: }

func.func @partial_overlap_starts_aligned(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<3x2xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
  %0 = stablehlo.dynamic_update_slice %cst, %arg0, %arg1, %arg2 : (tensor<32x32xf32>, tensor<4x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
  %1 = stablehlo.dynamic_slice %0, %arg1, %arg2, sizes = [3, 2] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK: func.func @partial_overlap_starts_aligned(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<3x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:3, 0:2] : (tensor<4x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:   return %0 : tensor<3x2xf32>
// CHECK-NEXT: }

func.func @partial_overlap_starts_not_aligned(%arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> (tensor<3x2xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
  %idx1 = stablehlo.constant dense<4> : tensor<i32>
  %idx2 = stablehlo.constant dense<7> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %cst, %arg0, %idx1, %arg1 : (tensor<32x32xf32>, tensor<8x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
  %1 = stablehlo.dynamic_slice %0, %idx2, %arg1, sizes = [3, 2] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK: func.func @partial_overlap_starts_not_aligned(%arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> tensor<3x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [3:6, 0:2] : (tensor<8x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:   return %0 : tensor<3x2xf32>
// CHECK-NEXT: }

func.func @partial_overlap_starts_not_aligned_pad(%arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> (tensor<3x2xf32>) {
  %cst = stablehlo.constant dense<2.000000e+00> : tensor<32x32xf32>
  %idx1 = stablehlo.constant dense<4> : tensor<i32>
  %idx2 = stablehlo.constant dense<2> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %cst, %arg0, %idx1, %arg1 : (tensor<32x32xf32>, tensor<8x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
  %1 = stablehlo.dynamic_slice %0, %idx2, %arg1, sizes = [3, 2] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK: func.func @partial_overlap_starts_not_aligned_pad(%arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> tensor<3x2xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:1, 0:2] : (tensor<8x3xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:   %1 = stablehlo.pad %0, %cst, low = [2, 0], high = [0, 0], interior = [0, 0] : (tensor<1x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK-NEXT:   return %1 : tensor<3x2xf32>
// CHECK-NEXT: }

func.func @partial_overlap_starts_not_aligned_pad2(%arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> (tensor<9x2xf32>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<32x32xf32>
  %idx1 = stablehlo.constant dense<4> : tensor<i32>
  %idx2 = stablehlo.constant dense<5> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %cst, %arg0, %idx1, %arg1 : (tensor<32x32xf32>, tensor<8x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
  %1 = stablehlo.dynamic_slice %0, %idx2, %arg1, sizes = [9, 2] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<9x2xf32>
  return %1 : tensor<9x2xf32>
}

// CHECK: func.func @partial_overlap_starts_not_aligned_pad2(%arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> tensor<9x2xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [1:8, 0:2] : (tensor<8x3xf32>) -> tensor<7x2xf32>
// CHECK-NEXT:   %1 = stablehlo.pad %0, %cst, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<7x2xf32>, tensor<f32>) -> tensor<9x2xf32>
// CHECK-NEXT:   return %1 : tensor<9x2xf32>
// CHECK-NEXT: }

func.func @partial_overlap_starts_not_aligned_cant_simplify(%input: tensor<32x32xf32>, %arg0: tensor<8x3xf32>, %arg1: tensor<i32>) -> (tensor<9x2xf32>) {
  %idx1 = stablehlo.constant dense<4> : tensor<i32>
  %idx2 = stablehlo.constant dense<5> : tensor<i32>
  // CHECK: stablehlo.dynamic_update_slice
  %0 = stablehlo.dynamic_update_slice %input, %arg0, %idx1, %arg1 : (tensor<32x32xf32>, tensor<8x3xf32>, tensor<i32>, tensor<i32>) -> tensor<32x32xf32>
  %1 = stablehlo.dynamic_slice %0, %idx2, %arg1, sizes = [9, 2] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<9x2xf32>
  return %1 : tensor<9x2xf32>
}
