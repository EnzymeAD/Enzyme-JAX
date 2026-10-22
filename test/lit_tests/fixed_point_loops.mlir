// RUN: enzymexlamlir-opt %s --raise-fixed-point-loops --split-input-file | FileCheck %s --check-prefix=RAISE
// RUN: enzymexlamlir-opt %s --raise-fixed-point-loops --lower-fixed-point-loops --split-input-file | FileCheck %s --check-prefix=LOWER

// A marked stablehlo.while moves over as it is.
func.func @hlo(%u0: tensor<3xf64>, %p: tensor<3xf64>) -> tensor<3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cap = stablehlo.constant dense<100> : tensor<i64>
  %r:3 = stablehlo.while(%i = %c0, %u = %u0, %pp = %p) : tensor<i64>, tensor<3xf64>, tensor<3xf64> attributes {enzyme.fixed_point, enzyme.fixed_point_max_iters = 50 : i64, other.attr}
   cond {
    %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %s = stablehlo.sine %u : tensor<3xf64>
    %un = stablehlo.add %s, %pp : tensor<3xf64>
    %in = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in, %un, %pp : tensor<i64>, tensor<3xf64>, tensor<3xf64>
  }
  return %r#1 : tensor<3xf64>
}

// RAISE-LABEL: func.func @hlo
// RAISE:      %[[R:.+]]:3 = enzymexla.fixed_point(%[[I:.+]] = %{{.+}}, %[[U:.+]] = %arg0, %[[P:.+]] = %arg1) : tensor<i64>, tensor<3xf64>, tensor<3xf64> attributes {max_iters = 50 : i64, other.attr}
// RAISE-NEXT: cond {
// RAISE:        enzymexla.yield %{{.+}} : tensor<i1>
// RAISE:      } body {
// RAISE:        enzymexla.yield %{{.+}}, %{{.+}}, %[[P]] : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// RAISE:      return %[[R]]#1

// LOWER-LABEL: func.func @hlo
// LOWER:      stablehlo.while{{.*}} attributes {enzyme.fixed_point, enzyme.fixed_point_max_iters = 50 : i64, other.attr}
// LOWER-NOT:  enzymexla.fixed_point

// -----

// An explicit opt-out is not raised.
func.func @opted_out(%u0: tensor<3xf64>) -> tensor<3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cap = stablehlo.constant dense<100> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %u = %u0) : tensor<i64>, tensor<3xf64> attributes {enzyme.fixed_point = false}
   cond {
    %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %s = stablehlo.sine %u : tensor<3xf64>
    %in = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in, %s : tensor<i64>, tensor<3xf64>
  }
  return %r#1 : tensor<3xf64>
}

// RAISE-LABEL: func.func @opted_out
// RAISE-NOT:   enzymexla.fixed_point
// RAISE:       stablehlo.while{{.*}} attributes {enzyme.fixed_point = false}

// LOWER-LABEL: func.func @opted_out
// LOWER:       stablehlo.while{{.*}} attributes {enzyme.fixed_point = false}
