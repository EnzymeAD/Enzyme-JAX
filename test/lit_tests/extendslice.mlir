// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=extend_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s


// CHECK:     func.func @f(%arg0: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>) {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:2, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<2x1520x3056xf64>
// CHECK-NEXT:   %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<2x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %arg0 [0:3, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:   %3 = "enzymexla.extend"(%2) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<3x1520x3056xf64>) -> tensor<4x1520x3056xf64>
// CHECK-NEXT:   return %1, %3 : tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>
// CHECK-NEXT: }
func.func @f(%a: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>) {
  %b = "enzymexla.extend"(%a) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<5x1520x3056xf64>
  %c = stablehlo.slice %b [0:3, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<3x1520x3056xf64>
  %d = stablehlo.slice %b [0:4, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<4x1520x3056xf64>
  return %c, %d : tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>
}
