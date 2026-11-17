// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=1 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=EXP1
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | stablehlo-translate --interpret
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | stablehlo-translate --interpret
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2 dot-general-to-reduce=false" %s | stablehlo-translate --interpret

// CHECK-NOT: unrealized_conversion_cast
// CHECK-LABEL: func.func @pow_general
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: stablehlo.power {{.*}} : tensor<4xf64>
// EXP1-LABEL: func.func @pow_general
// EXP1: stablehlo.power {{.*}} : tensor<4xf32>

func.func @pow_general(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %0 = stablehlo.power %x, %y : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @pow_tenth
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: stablehlo.power {{.*}} : tensor<4xf64>
func.func @pow_tenth(%x: tensor<4xf64>) -> tensor<4xf64> {
  %cst = stablehlo.constant dense<0.1> : tensor<4xf64>
  %0 = stablehlo.power %x, %cst : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @pow_const_specials
// CHECK-NOT: unrealized_conversion_cast
func.func @pow_const_specials(%x: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) {
  %c2 = stablehlo.constant dense<2.0> : tensor<2xf64>
  %chalf = stablehlo.constant dense<0.5> : tensor<2xf64>
  %cm1 = stablehlo.constant dense<-1.0> : tensor<2xf64>
  %c0 = stablehlo.constant dense<0.0> : tensor<2xf64>
  %sq = stablehlo.power %x, %c2 : tensor<2xf64>
  %rt = stablehlo.power %x, %chalf : tensor<2xf64>
  %inv = stablehlo.power %x, %cm1 : tensor<2xf64>
  %one = stablehlo.power %x, %c0 : tensor<2xf64>
  return %sq, %rt, %inv, %one : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // 1. Reactant.jl#2892 case: x^0.1 at multi-float precision (far closer than f32!)
  %x1 = stablehlo.constant dense<[0.5, 1.0, 2.5, 10.0]> : tensor<4xf64>
  %e1 = stablehlo.constant dense<[0.93303299153680741, 1.0, 1.0959582263141855, 1.2589254117941673]> : tensor<4xf64>
  %r1 = func.call @pow_tenth(%x1) : (tensor<4xf64>) -> tensor<4xf64>
  "check.expect_almost_eq"(%r1, %e1) {tolerance = 1.0e-9 : f64} : (tensor<4xf64>, tensor<4xf64>) -> ()

  // 2. General power: negative base with even/odd integer exponents, zero base, zero exponent
  %x2 = stablehlo.constant dense<[-3.0, -3.0, 0.0, -2.5]> : tensor<4xf64>
  %y2 = stablehlo.constant dense<[2.0, 3.0, 0.1, 0.0]> : tensor<4xf64>
  %e2 = stablehlo.constant dense<[9.0, -27.0, 0.0, 1.0]> : tensor<4xf64>
  %r2 = func.call @pow_general(%x2, %y2) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  "check.expect_almost_eq"(%r2, %e2) {tolerance = 1.0e-8 : f64} : (tensor<4xf64>, tensor<4xf64>) -> ()

  // 3. Negative base with non-integer exponent -> NaN, and 0^(-2) -> +inf
  %x3 = stablehlo.constant dense<[-2.0, 0.0, 4.0, 0.0]> : tensor<4xf64>
  %y3 = stablehlo.constant dense<[0.1, -2.0, 1.5, 0.0]> : tensor<4xf64>
  %r3 = func.call @pow_general(%x3, %y3) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  %nan_slice = stablehlo.slice %r3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
  %is_nan = stablehlo.compare NE, %nan_slice, %nan_slice : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xi1>
  "check.expect_eq_const"(%is_nan) {value = dense<true> : tensor<1xi1>} : (tensor<1xi1>) -> ()
  %inf_slice = stablehlo.slice %r3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
  %pos_inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<1xf64>
  "check.expect_eq"(%inf_slice, %pos_inf) : (tensor<1xf64>, tensor<1xf64>) -> ()
  %finite_slice = stablehlo.slice %r3 [2:4] : (tensor<4xf64>) -> tensor<2xf64>
  %e3_finite = stablehlo.constant dense<[8.0, 1.0]> : tensor<2xf64>
  "check.expect_almost_eq"(%finite_slice, %e3_finite) {tolerance = 1.0e-9 : f64} : (tensor<2xf64>, tensor<2xf64>) -> ()

  // 4. Constant-exponent specializations (x^2, x^0.5, x^-1, x^0)
  %x4 = stablehlo.constant dense<[4.0, 9.0]> : tensor<2xf64>
  %sq, %rt, %inv, %one = func.call @pow_const_specials(%x4) : (tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>)
  %e_sq = stablehlo.constant dense<[16.0, 81.0]> : tensor<2xf64>
  %e_rt = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf64>
  %e_inv = stablehlo.constant dense<[0.25, 0.1111111111111111]> : tensor<2xf64>
  %e_one = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  "check.expect_almost_eq"(%sq, %e_sq) {tolerance = 1.0e-13 : f64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_almost_eq"(%rt, %e_rt) {tolerance = 1.0e-13 : f64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_almost_eq"(%inv, %e_inv) {tolerance = 1.0e-13 : f64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_almost_eq"(%one, %e_one) {tolerance = 1.0e-13 : f64} : (tensor<2xf64>, tensor<2xf64>) -> ()

  return
}
