// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%pred: tensor<i1>, %a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %pred, %a, %b : tensor<i1>, tensor<f32>
    %1 = "stablehlo.if"(%pred) ({
      %2 = stablehlo.add %0, %0 : tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }, {
      %3 = stablehlo.add %0, %0 : tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %0 = "stablehlo.if"(%arg0) ({
// CHECK-NEXT:      %1 = stablehlo.add %arg1, %arg1 : tensor<f32>
// CHECK-NEXT:      stablehlo.return %1 : tensor<f32>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %1 = stablehlo.add %arg2, %arg2 : tensor<f32>
// CHECK-NEXT:      stablehlo.return %1 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f32>
// CHECK-NEXT:    return %0 : tensor<f32>
// CHECK-NEXT:  }
