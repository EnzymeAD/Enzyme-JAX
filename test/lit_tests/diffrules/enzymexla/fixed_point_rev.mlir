// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize --lower-fixed-point-loops | FileCheck %s

// Reverse mode through an enzymexla.fixed_point with tensor state: the primal
// op runs untaped (no dynamic_update_slice), the adjoint is a stablehlo.while
// whose condition tests the reduction of the adjoint increment, and the primal
// lowers back to a stablehlo.while afterwards.

module {
  func.func private @solve(%u0: tensor<3xf64>, %p: tensor<3xf64>) -> tensor<3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %cap = stablehlo.constant dense<100> : tensor<i64>
    %tol = stablehlo.constant dense<1.0e-12> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %r:4 = enzymexla.fixed_point(%i = %c0, %u = %u0, %err = %one, %pp = %p) : tensor<i64>, tensor<3xf64>, tensor<f64>, tensor<3xf64> attributes {reduction = 1.0e-10 : f64, max_iters = 50 : i64}
     cond {
      %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %gt = stablehlo.compare GT, %err, %tol : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %go = stablehlo.and %lt, %gt : tensor<i1>
      enzymexla.yield %go : tensor<i1>
    } body {
      %s = stablehlo.sine %u : tensor<3xf64>
      %half = stablehlo.constant dense<0.5> : tensor<3xf64>
      %hs = stablehlo.multiply %half, %s : tensor<3xf64>
      %un = stablehlo.add %hs, %pp : tensor<3xf64>
      %d = stablehlo.subtract %un, %u : tensor<3xf64>
      %ad = stablehlo.abs %d : tensor<3xf64>
      %z = stablehlo.constant dense<0.0> : tensor<f64>
      %e = stablehlo.reduce(%ad init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
      %in = stablehlo.add %i, %c1 : tensor<i64>
      enzymexla.yield %in, %un, %e, %pp : tensor<i64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>
    }
    return %r#1 : tensor<3xf64>
  }

  func.func @main(%u0: tensor<3xf64>, %p: tensor<3xf64>, %seed: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
    %r:3 = enzyme.autodiff @solve(%u0, %p, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
    return %r#0, %r#1, %r#2 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }
}

// CHECK-LABEL: func.func private @diffesolve
// CHECK-NOT: dynamic_update_slice
// CHECK: stablehlo.while{{.*}} attributes {enzyme.fixed_point, enzyme.fixed_point_max_iters = 50 : i64, enzyme.fixed_point_reduction = 1.000000e-10 : f64}
// CHECK-NOT: dynamic_update_slice
// CHECK: stablehlo.while
// CHECK: cond {
// CHECK: stablehlo.compare GT
// CHECK: } do {
// CHECK: stablehlo.cosine
// CHECK-NOT: dynamic_slice
// CHECK: return
