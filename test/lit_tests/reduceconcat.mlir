// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2xf32>, %b : tensor<1xf32>, %c : tensor<1xf32>) -> tensor<f32> {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %a, %b, %c, dim=0 : (tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>

    %1308 = stablehlo.reduce(%concat init: %cst0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>

    return %1308 : tensor<f32>

  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<f32> {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %2 = stablehlo.add %0, %1 : tensor<f32>
// CHECK-NEXT:    %3 = stablehlo.reshape %arg2 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 : tensor<f32>
// CHECK-NEXT:    return %4 : tensor<f32>
// CHECK-NEXT:  }
