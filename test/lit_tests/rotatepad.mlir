// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%6548: tensor<4x6x80xf64>, %cst_286 : tensor<f64>) -> tensor<4x7x80xf64> {
      %6549 = stablehlo.pad %6548, %cst_286, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x6x80xf64>, tensor<f64>) -> tensor<4x7x80xf64>
      %11824 = "enzymexla.rotate"(%6549) <{amount = 79 : si32, dimension = 2 : si32}> : (tensor<4x7x80xf64>) -> tensor<4x7x80xf64>
      stablehlo.return %11824 : tensor<4x7x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x6x80xf64>, %arg1: tensor<f64>) -> tensor<4x7x80xf64> {
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 79 : si32, dimension = 2 : si32}> : (tensor<4x6x80xf64>) -> tensor<4x6x80xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x6x80xf64>, tensor<f64>) -> tensor<4x7x80xf64>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x7x80xf64>
// CHECK-NEXT:  }