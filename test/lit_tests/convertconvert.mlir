// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<4xf32>) -> tensor<4xf32> {
    %conv0 = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xbf16>
    %conv = stablehlo.convert %conv0 : (tensor<4xbf16>) -> tensor<4xf32>
    return %conv : tensor<4xf32>
  }

  func.func @main2(%a : tensor<4xf64>) -> tensor<4xbf16> {
    %conv0 = stablehlo.convert %a : (tensor<4xf64>) -> tensor<4xf32>
    %conv = stablehlo.convert %conv0 : (tensor<4xf32>) -> tensor<4xbf16>
    return %conv : tensor<4xbf16>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    return %arg0 : tensor<4xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<4xf64>) -> tensor<4xbf16> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xbf16>
// CHECK-NEXT:    return %0 : tensor<4xbf16>
// CHECK-NEXT:  }
