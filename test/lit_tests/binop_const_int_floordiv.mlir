// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=binop_const_simplify},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// The MulDivConst / DivMulConst constant-lifting patterns reassociate a
// division with a multiplication: (x / a) * b -> x * (b / a) and
// (x * a) / b -> x * (a / b). This is only valid for floating point.
// For integers the division is a floor-division, so reassociating it through
// the multiplication is incorrect, e.g. (x / 16) * 2 != x * (2 / 16) == x * 0.
// These patterns must NOT fire on integer element types.

// CHECK-LABEL: func @muldivconst_int
// CHECK:      %[[D:.*]] = stablehlo.divide %arg0, %{{.*}} : tensor<32xi64>
// CHECK-NEXT: %[[M:.*]] = stablehlo.multiply %[[D]], %{{.*}} : tensor<32xi64>
// CHECK-NEXT: return %[[M]]
func.func @muldivconst_int(%arg0: tensor<32xi64>) -> tensor<32xi64> {
  %c16 = stablehlo.constant dense<16> : tensor<32xi64>
  %c2 = stablehlo.constant dense<2> : tensor<32xi64>
  %0 = stablehlo.divide %arg0, %c16 : tensor<32xi64>
  %1 = stablehlo.multiply %0, %c2 : tensor<32xi64>
  return %1 : tensor<32xi64>
}

// CHECK-LABEL: func @divmulconst_int
// CHECK:      %[[M:.*]] = stablehlo.multiply %arg0, %{{.*}} : tensor<32xi64>
// CHECK-NEXT: %[[D:.*]] = stablehlo.divide %[[M]], %{{.*}} : tensor<32xi64>
// CHECK-NEXT: return %[[D]]
func.func @divmulconst_int(%arg0: tensor<32xi64>) -> tensor<32xi64> {
  %c16 = stablehlo.constant dense<16> : tensor<32xi64>
  %c2 = stablehlo.constant dense<2> : tensor<32xi64>
  %0 = stablehlo.multiply %arg0, %c2 : tensor<32xi64>
  %1 = stablehlo.divide %0, %c16 : tensor<32xi64>
  return %1 : tensor<32xi64>
}

// Floating point is still reassociated: (x / 4) * 2 -> x * (2 / 4).
// CHECK-LABEL: func @muldivconst_float
// CHECK:      %[[C:.*]] = stablehlo.divide %{{.*}}, %{{.*}} : tensor<f64>
// CHECK-NEXT: %[[R:.*]] = stablehlo.multiply %arg0, %[[C]] : tensor<f64>
// CHECK-NEXT: return %[[R]]
func.func @muldivconst_float(%arg0: tensor<f64>) -> tensor<f64> {
  %c2 = stablehlo.constant dense<2.0> : tensor<f64>
  %c4 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.divide %arg0, %c4 : tensor<f64>
  %1 = stablehlo.multiply %0, %c2 : tensor<f64>
  return %1 : tensor<f64>
}
