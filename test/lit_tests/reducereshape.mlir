// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<20xf32>, %b : tensor<20xf32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %ar = stablehlo.reshape %a : (tensor<20xf32>) -> tensor<4x5xf32>
    %br = stablehlo.reshape %b : (tensor<20xf32>) -> tensor<4x5xf32>

    %ma = stablehlo.add %ar, %br : tensor<4x5xf32>
    %mb = stablehlo.multiply %ma, %ma : tensor<4x5xf32>


    %1308 = stablehlo.reduce(%mb init: %c0) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x5xf32>, tensor<f32>) -> tensor<f32>

    return %1308 : tensor<f32>

  }
}

// CHECK:  func.func @main(%arg0: tensor<20xf32>, %arg1: tensor<20xf32>) -> tensor<f32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.add %arg0, %arg1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<20xf32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0] : (tensor<20xf32>, tensor<20xf32>) -> tensor<f32>
// CHECK-NEXT:    return %[[i1]] : tensor<f32>
// CHECK-NEXT:  }
