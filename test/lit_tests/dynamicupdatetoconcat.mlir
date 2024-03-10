// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%a : tensor<4x5xf32>, %b : tensor<2x5xf32>) -> tensor<4x5xf32> {
      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %c0 = stablehlo.constant dense<0> : tensor<i32>
      %r = stablehlo.dynamic_update_slice %a, %b, %c1, %c0 : (tensor<4x5xf32>, tensor<2x5xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    return %r : tensor<4x5xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<2x5xf32>) -> tensor<4x5xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:5] : (tensor<4x5xf32>) -> tensor<1x5xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [3:4, 0:5] : (tensor<4x5xf32>) -> tensor<1x5xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg1, %1, dim = 0 : (tensor<1x5xf32>, tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:    return %2 : tensor<4x5xf32>
// CHECK-NEXT:  }
