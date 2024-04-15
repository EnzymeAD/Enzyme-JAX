// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main() -> tensor<1xf32> {
    %concat = stablehlo.constant dense<3.140000e+00> : tensor<f32>
    %conv = stablehlo.reshape %concat : (tensor<f32>) -> tensor<1xf32>
    return %conv : tensor<1xf32>
  }
}

// CHECK:  func.func @main() -> tensor<1xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.140000e+00> : tensor<1xf32>
// CHECK-NEXT:    return %[[i0]] : tensor<1xf32>
// CHECK-NEXT:  }
