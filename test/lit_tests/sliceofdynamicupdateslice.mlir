// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<4x5xf32>, %b : tensor<4x5xf32>) -> tensor<2x2xf32> {

      %c2 = stablehlo.constant dense<2> : tensor<i32>
      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %r = stablehlo.dynamic_update_slice %a, %b, %c2, %c1 : (tensor<4x5xf32>, tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
      %slice = stablehlo.slice %r [2:4, 1:5:2] : (tensor<4x5xf32>) -> tensor<2x2xf32>
    return %slice : tensor<2x2xf32>
  }

  func.func @skip(%a : tensor<4x5xf32>, %b : tensor<4x5xf32>) -> tensor<1x2xf32> {

      %c2 = stablehlo.constant dense<2> : tensor<i32>
      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %r = stablehlo.dynamic_update_slice %a, %b, %c2, %c1 : (tensor<4x5xf32>, tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
      %slice = stablehlo.slice %r [1:2, 1:5:2] : (tensor<4x5xf32>) -> tensor<1x2xf32>
    return %slice : tensor<1x2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:2, 0:4:2] : (tensor<4x5xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    return %0 : tensor<2x2xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @skip(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:2, 1:5:2] : (tensor<4x5xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:    return %0 : tensor<1x2xf32>
// CHECK-NEXT:  }
