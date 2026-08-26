// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<5x4xf32>, %b : tensor<5x4xf32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %ar = stablehlo.transpose %a, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
    %br = stablehlo.transpose %b, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>

    %ma = stablehlo.add %ar, %br : tensor<4x5xf32>
    %mb = stablehlo.multiply %ma, %ma : tensor<4x5xf32>


    %1308 = stablehlo.reduce(%mb init: %c0) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x5xf32>, tensor<f32>) -> tensor<f32>

    return %1308 : tensor<f32>

  }
}

// CHECK:  func.func @main(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<f32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.add %arg0, %arg1 : tensor<5x4xf32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.dot_general %[[i0]], %[[i0]], contracting_dims = [0, 1] x [0, 1] : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<f32>
// CHECK-NEXT:    %[[i1]] : tensor<f32>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<3x4x5x6xf32>) -> tensor<6x4xf32> {
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<3x4x5x6xf32>) -> tensor<3x6x4x5xf32>
    %1 = stablehlo.reduce(%0 init: %cst_2) applies stablehlo.add across dimensions = [0, 3] : (tensor<3x6x4x5xf32>, tensor<f32>) -> tensor<6x4xf32>
    return %1 : tensor<6x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3x4x5x6xf32>) -> tensor<6x4xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reduce(%arg0 init: %[[i0]]) applies stablehlo.add across dimensions = [0, 2] : (tensor<3x4x5x6xf32>, tensor<f32>) -> tensor<4x6xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.transpose %[[i1]], dims = [1, 0] : (tensor<4x6xf32>) -> tensor<6x4xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<6x4xf32>
// CHECK-NEXT:  }
