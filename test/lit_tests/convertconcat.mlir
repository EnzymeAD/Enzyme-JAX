// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2xf32>, %b : tensor<1xf32>, %c : tensor<1xf32>) -> tensor<4xbf16> {
    %concat = stablehlo.concatenate %a, %b, %c, dim=0 : (tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
    %conv = stablehlo.convert %concat : (tensor<4xf32>) -> tensor<4xbf16>
    return %conv : tensor<4xbf16>

  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<4xbf16> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xbf16>
// CHECK-NEXT:    %1 = stablehlo.convert %arg1 : (tensor<1xf32>) -> tensor<1xbf16>
// CHECK-NEXT:    %2 = stablehlo.convert %arg2 : (tensor<1xf32>) -> tensor<1xbf16>
// CHECK-NEXT:    %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<2xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<4xbf16>
// CHECK-NEXT:    return %3 : tensor<4xbf16>
// CHECK-NEXT:  }
