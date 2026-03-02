// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=extend_pad --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%77: tensor<4x3055x6128xf64>, %cst: tensor<f64>) -> tensor<6x3056x6128xf64> {


    %78 = stablehlo.pad %77, %cst, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x3055x6128xf64>, tensor<f64>) -> tensor<4x3056x6128xf64>

    %82 = "enzymexla.extend"(%78) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x3056x6128xf64>) -> tensor<6x3056x6128xf64>

    return %82 : tensor<6x3056x6128xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x3055x6128xf64>, %arg1: tensor<f64>) -> tensor<6x3056x6128xf64> {
// CHECK-NEXT:    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x3055x6128xf64>) -> tensor<6x3055x6128xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<6x3055x6128xf64>, tensor<f64>) -> tensor<6x3056x6128xf64>
// CHECK-NEXT:    return %1 : tensor<6x3056x6128xf64>
// CHECK-NEXT:  }
