// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main() -> tensor<1x4x1x5x1xf32> {
    %iot = stablehlo.iota dim=1 : tensor<4x5xf32>
    %conv = stablehlo.reshape %iot : (tensor<4x5xf32>) -> tensor<1x4x1x5x1xf32>
    return %conv : tensor<1x4x1x5x1xf32>
  }
}

// CHECK:  func.func @main() -> tensor<1x4x1x5x1xf32> {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 3 : tensor<1x4x1x5x1xf32>
// CHECK-NEXT:    return %0 : tensor<1x4x1x5x1xf32>
// CHECK-NEXT:  }
