// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<4x5xf32>) -> tensor<2x2xf32> {

      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %c2 = stablehlo.constant dense<2> : tensor<i32>
      %r = stablehlo.dynamic_slice %a, %c1, %c2, sizes = [2, 2] : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
    return %r : tensor<2x2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x5xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:3, 2:4] : (tensor<4x5xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    return %0 : tensor<2x2xf32>
// CHECK-NEXT:  }
