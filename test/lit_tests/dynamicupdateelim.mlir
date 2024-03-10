// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<4x5xf32>, %b : tensor<4x5xf32>) -> tensor<4x5xf32> {

      %c0 = stablehlo.constant dense<0> : tensor<i32>
      %r = stablehlo.dynamic_update_slice %a, %b, %c0, %c0 : (tensor<4x5xf32>, tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    return %r : tensor<4x5xf32>
  }
}

// CHECK:   func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
// CHECK-NEXT:     return %arg1 : tensor<4x5xf32>
// CHECK-NEXT:   }
