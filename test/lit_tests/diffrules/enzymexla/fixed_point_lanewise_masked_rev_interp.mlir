// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,canonicalize,arith-raise{stablehlo=true},raise-fixed-point-loops,enzyme{postpasses=arith-raise{stablehlo=true},canonicalize,remove-unnecessary-enzyme-ops,canonicalize},lower-fixed-point-loops,canonicalize)" | stablehlo-translate - --interpret --allow-unregistered-dialect

// Reverse mode through a per-thread Newton iteration for sqrt(a), marked as a
// fixed point and raised with a lane mask. Every lane has stopped at the
// fixed point, so the masked step there is the identity; the reverse rule has
// to differentiate the step with every lane active to get
// d sqrt(a)/da = 1/(2 sqrt(a)). Lane 0 has converged before the first step and
// lane 4 takes dozens.

module {
  func.func private @newton(%x: memref<5xf64, 1>, %a: memref<5xf64, 1>) {
    %two = arith.constant 2.0 : f64
    %tol = arith.constant 1.0e-14 : f64
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %cap = arith.constant 500 : i64
    affine.parallel (%i) = (0) to (5) {
      %ai = affine.load %a[%i] : memref<5xf64, 1>
      %r:2 = scf.while (%xi = %ai, %k = %c0) : (f64, i64) -> (f64, i64) {
        %xx = arith.mulf %xi, %xi : f64
        %f = arith.subf %xx, %ai : f64
        %d = arith.mulf %two, %xi : f64
        %q = arith.divf %f, %d : f64
        %aq = math.absf %q : f64
        %ax = math.absf %xi : f64
        %t = arith.mulf %ax, %tol : f64
        %moving = arith.cmpf ogt, %aq, %t : f64
        %capped = arith.cmpi slt, %k, %cap : i64
        %go = arith.andi %moving, %capped : i1
        scf.condition(%go) %xi, %k : f64, i64
      } do {
      ^bb0(%xi: f64, %k: i64):
        %xx = arith.mulf %xi, %xi : f64
        %f = arith.subf %xx, %ai : f64
        %d = arith.mulf %two, %xi : f64
        %q = arith.divf %f, %d : f64
        %xn = arith.subf %xi, %q : f64
        %kn = arith.addi %k, %c1 : i64
        scf.yield %xn, %kn : f64, i64
      } attributes {enzyme.fixed_point}
      affine.store %r#0, %x[%i] : memref<5xf64, 1>
    }
    return
  }

  func.func @loss(%a: tensor<5xf64>) -> tensor<f64> {
    %x0 = stablehlo.constant dense<0.0> : tensor<5xf64>
    %x = enzymexla.jit_call @newton (%x0, %a) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<5xf64>, tensor<5xf64>) -> tensor<5xf64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %s = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<5xf64>, tensor<f64>) -> tensor<f64>
    return %s : tensor<f64>
  }

  func.func @main() {
    %a = stablehlo.constant dense<[1.0, 4.0, 2.0, 9.0, 1.0e12]> : tensor<5xf64>
    %seed = stablehlo.constant dense<1.0> : tensor<f64>
    %r:2 = enzyme.autodiff @loss(%a, %seed) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<5xf64>, tensor<f64>) -> (tensor<f64>, tensor<5xf64>)
    // sum(sqrt(a))
    check.expect_almost_eq_const %r#0, dense<1000007.4142135624> : tensor<f64>
    // 1 / (2 sqrt(a))
    check.expect_almost_eq_const %r#1, dense<[0.5, 0.25, 0.35355339059327373, 0.16666666666666666, 5.0e-7]> : tensor<5xf64>
    return
  }
}
