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
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.add %arg0, %arg1 : tensor<20xf32>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %1 : tensor<20xf32>
// CHECK-NEXT:    %3 = stablehlo.reduce(%2 init: %0) applies stablehlo.add across dimensions = [0] : (tensor<20xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %3 : tensor<f32>
// CHECK-NEXT:  }
