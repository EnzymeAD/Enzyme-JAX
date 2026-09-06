// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=extend_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// CHECK:     func.func @f_single_use(%arg0: tensor<4x1520x3056xf64>) -> tensor<3x1520x3056xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:2, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<2x1520x3056xf64>
// CHECK-NEXT:   %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<2x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:   return %1 : tensor<3x1520x3056xf64>
// CHECK-NEXT: }
func.func @f_single_use(%a: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>) {
  %b = "enzymexla.extend"(%a) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<5x1520x3056xf64>
  %c = stablehlo.slice %b [0:3, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<3x1520x3056xf64>
  return %c : tensor<3x1520x3056xf64>
}


// CHECK:      func.func @f_multiple_uses(%arg0: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>) {
// CHECK-NEXT:   %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<5x1520x3056xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:3, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %0 [0:4, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<4x1520x3056xf64>
// CHECK-NEXT:   return %1, %2 : tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>
// CHECK-NEXT: }
func.func @f_multiple_uses(%a: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>) {
  %b = "enzymexla.extend"(%a) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<5x1520x3056xf64>
  %c = stablehlo.slice %b [0:3, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<3x1520x3056xf64>
  %d = stablehlo.slice %b [0:4, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<4x1520x3056xf64>
  return %c, %d : tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>
}

// CHECK:      func.func @f_multiple_uses_superfluous_extend(%arg0: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>) {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:3, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:   %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<3x1520x3056xf64>) -> tensor<4x1520x3056xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %arg0 [0:3, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<3x1520x3056xf64>
// CHECK-NEXT:   return %2, %1 : tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>
// CHECK-NEXT: }
func.func @f_multiple_uses_superfluous_extend(%a: tensor<4x1520x3056xf64>) -> (tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>) {
  %b = "enzymexla.extend"(%a) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<5x1520x3056xf64>
  %c = stablehlo.slice %b [0:4, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<4x1520x3056xf64>
  %d = stablehlo.slice %b [1:4, 0:1520, 0:3056] : (tensor<5x1520x3056xf64>) -> tensor<3x1520x3056xf64>
  return %d, %c : tensor<3x1520x3056xf64>, tensor<4x1520x3056xf64>
}
