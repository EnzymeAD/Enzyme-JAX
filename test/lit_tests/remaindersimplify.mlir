// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose(%4422: tensor<ui32>) -> tensor<ui32> {
  %c_199 = stablehlo.constant dense<1> : tensor<ui32>
  %4423 = stablehlo.remainder %4422, %c_199 : tensor<ui32>
  return %4423 : tensor<ui32>
}

// CHECK:  func.func @transpose(%arg0: tensor<ui32>) -> tensor<ui32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:    return %c : tensor<ui32>
// CHECK-NEXT:  }
