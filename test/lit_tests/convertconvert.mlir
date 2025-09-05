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

  func.func @main3(%arg0: tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi16> {
    %0 = stablehlo.convert %arg0 : (tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi64>
    %1 = stablehlo.convert %0 : (tensor<1x2x3x20xi64>) -> tensor<1x2x3x20xi16>
    return %1 : tensor<1x2x3x20xi16>
  }

  func.func @fail1(%arg0: tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi64> {
    %0 = stablehlo.convert %arg0 : (tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi4>
    %1 = stablehlo.convert %0 : (tensor<1x2x3x20xi4>) -> tensor<1x2x3x20xi64>
    return %1 : tensor<1x2x3x20xi64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    return %arg0 : tensor<4xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<4xf64>) -> tensor<4xbf16> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xbf16>
// CHECK-NEXT:    return %0 : tensor<4xbf16>
// CHECK-NEXT:  }
// CHECK: func.func @main3(%arg0: tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi16> {
// CHECK-NEXT:     %0 = stablehlo.convert %arg0 : (tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi16>
// CHECK-NEXT:     return %0 : tensor<1x2x3x20xi16>
// CHECK-NEXT: }
// CHECK: func.func @fail1(%arg0: tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi64> {
// CHECK-NEXT:     %0 = stablehlo.convert %arg0 : (tensor<1x2x3x20xi32>) -> tensor<1x2x3x20xi4>
// CHECK-NEXT:     %1 = stablehlo.convert %0 : (tensor<1x2x3x20xi4>) -> tensor<1x2x3x20xi64>
// CHECK-NEXT:     return %1 : tensor<1x2x3x20xi64>
// CHECK-NEXT: }
