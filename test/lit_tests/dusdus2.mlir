// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @fuse(%op: tensor<4x1x8x8xcomplex<f32>>, %19: tensor<4x1x4x8xcomplex<f32>>, %22: tensor<4x1x4x8xcomplex<f32>>) -> tensor<4x1x8x8xcomplex<f32>> {
    %c_7 = stablehlo.constant dense<4> : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.dynamic_update_slice %op, %19, %c_6, %c_6, %c_6, %c_6 : (tensor<4x1x8x8xcomplex<f32>>, tensor<4x1x4x8xcomplex<f32>>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xcomplex<f32>>
    %26 = stablehlo.dynamic_update_slice %23, %22, %c_6, %c_6, %c_6, %c_7 : (tensor<4x1x8x8xcomplex<f32>>, tensor<4x1x4x8xcomplex<f32>>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xcomplex<f32>>
    func.return %26: tensor<4x1x8x8xcomplex<f32>>
  }
}

// CHECK:  func.func @fuse(%arg0: tensor<4x1x8x8xcomplex<f32>>, %arg1: tensor<4x1x4x8xcomplex<f32>>, %arg2: tensor<4x1x4x8xcomplex<f32>>) -> tensor<4x1x8x8xcomplex<f32>> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<4> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg1, %arg2, %c, %c, %c, %c_0 : (tensor<4x1x4x8xcomplex<f32>>, tensor<4x1x4x8xcomplex<f32>>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x4x8xcomplex<f32>>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:4, 0:1, 4:8, 0:8] : (tensor<4x1x8x8xcomplex<f32>>) -> tensor<4x1x4x8xcomplex<f32>>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<4x1x4x8xcomplex<f32>>, tensor<4x1x4x8xcomplex<f32>>) -> tensor<4x1x8x8xcomplex<f32>>
// CHECK-NEXT:    return %2 : tensor<4x1x8x8xcomplex<f32>>
// CHECK-NEXT:  }
