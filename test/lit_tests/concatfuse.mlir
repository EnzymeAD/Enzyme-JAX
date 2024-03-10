// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2xf32>, %b : tensor<1xf32>, %c : tensor<1xf32>) -> tensor<4xf32> {
    %concat = stablehlo.concatenate %a, %b, dim=0 : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
    %concat2 = stablehlo.concatenate %concat, %c, dim=0 : (tensor<3xf32>, tensor<1xf32>) -> tensor<4xf32>
    return %concat2 : tensor<4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %[[concat:.+]] = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 0 : (tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[concat]] : tensor<4xf32>
// CHECK-NEXT:  }
