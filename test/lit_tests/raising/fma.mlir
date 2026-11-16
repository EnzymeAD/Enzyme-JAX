// RUN: enzymexlamlir-opt %s --arith-raise | FileCheck %s

func.func @f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = math.fma %arg0, %arg1, %arg2 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @f64(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>, %arg2: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %0 = math.fma %arg0, %arg1, %arg2 : tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}

func.func @zero_dim(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = math.fma %arg0, %arg1, %arg2 : tensor<f32>
  return %0 : tensor<f32>
}

func.func @scalar(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  %0 = math.fma %arg0, %arg1, %arg2 : f32
  return %0 : f32
}

// CHECK:  func.func @f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.fma %arg0, %arg1, %arg2 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @f64(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>, %arg2: tensor<2x3xf64>) -> tensor<2x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.fma %arg0, %arg1, %arg2 : tensor<2x3xf64>
// CHECK-NEXT:    return %0 : tensor<2x3xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @zero_dim(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %0 = stablehlo.fma %arg0, %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:    return %0 : tensor<f32>
// CHECK-NEXT:  }

// CHECK:  func.func @scalar(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
// CHECK-NEXT:    %0 = math.fma %arg0, %arg1, %arg2 : f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }
