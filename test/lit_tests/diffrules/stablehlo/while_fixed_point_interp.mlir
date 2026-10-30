// RUN: enzymexlamlir-opt %s --pass-pipeline="any(raise-fixed-point-loops,enzyme{postpasses=arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize,arith-raise{stablehlo=true} verifyPostPasses=true},inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },enzyme-hlo-opt,remove-unnecessary-enzyme-ops,lower-fixed-point-loops)" | stablehlo-translate - --interpret --allow-unregistered-dialect

// End to end from the attribute: a stablehlo.while marked enzyme.fixed_point
// (as `@trace fixed_point=...` emits it) is raised to the op, differentiated,
// and lowered back.
//
// The fixed point u = c sin(u) + p^2, iterated under a mixed condition (a
// residual test and an iteration cap), with the parameter p threaded through
// the loop and c read from above it. Differentiating the loop marked as a
// fixed point (one adjoint iteration at the converged state) must give the
// gradient of differentiating through every iteration of a counted loop that
// runs the same step to convergence, and the implicit function theorem.
//
// The reduction is on the squared norm of the adjoint increment, so the
// default 1e-12 leaves a relative error of ~1e-6 in the gradient (the tail of
// the geometric series). 1e-24 brings it to the 1e-8 the checks ask for.

module {
  func.func private @loss_marked(%u0: tensor<3xf64>, %p: tensor<3xf64>, %c: tensor<f64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %cap = stablehlo.constant dense<500> : tensor<i64>
    %tol = stablehlo.constant dense<1.0e-15> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %cb = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r:4 = stablehlo.while(%i = %c0, %u = %u0, %err = %one, %pp = %p) : tensor<i64>, tensor<3xf64>, tensor<f64>, tensor<3xf64> attributes {enzyme.fixed_point, enzyme.fixed_point_reduction = 1.0e-24 : f64}
     cond {
      %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %gt = stablehlo.compare GT, %err, %tol : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %go = stablehlo.and %lt, %gt : tensor<i1>
      stablehlo.return %go : tensor<i1>
    } do {
      %s = stablehlo.sine %u : tensor<3xf64>
      %cs = stablehlo.multiply %cb, %s : tensor<3xf64>
      %p2 = stablehlo.multiply %pp, %pp : tensor<3xf64>
      %un = stablehlo.add %cs, %p2 : tensor<3xf64>
      %d = stablehlo.subtract %un, %u : tensor<3xf64>
      %ad = stablehlo.abs %d : tensor<3xf64>
      %z = stablehlo.constant dense<0.0> : tensor<f64>
      %e = stablehlo.reduce(%ad init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
      %in = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %in, %un, %e, %pp : tensor<i64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>
    }
    %u3 = stablehlo.multiply %r#1, %r#1 : tensor<3xf64>
    %u3b = stablehlo.multiply %u3, %r#1 : tensor<3xf64>
    %z0 = stablehlo.constant dense<0.0> : tensor<f64>
    %l = stablehlo.reduce(%u3b init: %z0) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %l : tensor<f64>
  }

  func.func private @loss_plain(%u0: tensor<3xf64>, %p: tensor<3xf64>, %c: tensor<f64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %cap = stablehlo.constant dense<500> : tensor<i64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %cb = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r:4 = stablehlo.while(%i = %c0, %u = %u0, %err = %one, %pp = %p) : tensor<i64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>
     cond {
      %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %s = stablehlo.sine %u : tensor<3xf64>
      %cs = stablehlo.multiply %cb, %s : tensor<3xf64>
      %p2 = stablehlo.multiply %pp, %pp : tensor<3xf64>
      %un = stablehlo.add %cs, %p2 : tensor<3xf64>
      %d = stablehlo.subtract %un, %u : tensor<3xf64>
      %ad = stablehlo.abs %d : tensor<3xf64>
      %z = stablehlo.constant dense<0.0> : tensor<f64>
      %e = stablehlo.reduce(%ad init: %z) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
      %in = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %in, %un, %e, %pp : tensor<i64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>
    }
    %u3 = stablehlo.multiply %r#1, %r#1 : tensor<3xf64>
    %u3b = stablehlo.multiply %u3, %r#1 : tensor<3xf64>
    %z0 = stablehlo.constant dense<0.0> : tensor<f64>
    %l = stablehlo.reduce(%u3b init: %z0) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %l : tensor<f64>
  }

  func.func @main() {
    %u0 = stablehlo.constant dense<[0.1, 0.2, 0.3]> : tensor<3xf64>
    %p = stablehlo.constant dense<[0.5, 0.7, 0.9]> : tensor<3xf64>
    %c = stablehlo.constant dense<0.3> : tensor<f64>
    %seed = stablehlo.constant dense<1.0> : tensor<f64>

    %marked:3 = enzyme.autodiff @loss_marked(%u0, %p, %c, %seed) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<3xf64>, tensor<3xf64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<3xf64>, tensor<f64>)
    %plain:3 = enzyme.autodiff @loss_plain(%u0, %p, %c, %seed) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<3xf64>, tensor<3xf64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<3xf64>, tensor<f64>)

    // Same primal, same gradients for p (threaded through the loop) and c
    // (read from above it).
    check.expect_almost_eq %marked#0, %plain#0, tolerance = 1.0e-10 : tensor<f64>
    check.expect_almost_eq %marked#1, %plain#1, tolerance = 1.0e-8 : tensor<3xf64>
    check.expect_almost_eq %marked#2, %plain#2, tolerance = 1.0e-8 : tensor<f64>

    // And both agree with the implicit function theorem:
    //   dL/dp_i = 3 u_i^2 * 2 p_i / (1 - c cos u_i)
    //   dL/dc   = sum_i 3 u_i^2 sin(u_i) / (1 - c cos u_i)
    // evaluated at the fixed point found by the primal.
    %u = stablehlo.constant dense<[0.1, 0.2, 0.3]> : tensor<3xf64>
    %cb = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %ustar = call @fixed_point(%u, %p, %cb) : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %three = stablehlo.constant dense<3.0> : tensor<3xf64>
    %two = stablehlo.constant dense<2.0> : tensor<3xf64>
    %onev = stablehlo.constant dense<1.0> : tensor<3xf64>
    %uu = stablehlo.multiply %ustar, %ustar : tensor<3xf64>
    %three_uu = stablehlo.multiply %three, %uu : tensor<3xf64>
    %cosu = stablehlo.cosine %ustar : tensor<3xf64>
    %ccos = stablehlo.multiply %cb, %cosu : tensor<3xf64>
    %den = stablehlo.subtract %onev, %ccos : tensor<3xf64>
    %twop = stablehlo.multiply %two, %p : tensor<3xf64>
    %num_p = stablehlo.multiply %three_uu, %twop : tensor<3xf64>
    %dp = stablehlo.divide %num_p, %den : tensor<3xf64>
    check.expect_almost_eq %marked#1, %dp, tolerance = 1.0e-8 : tensor<3xf64>
    %sinu = stablehlo.sine %ustar : tensor<3xf64>
    %num_c = stablehlo.multiply %three_uu, %sinu : tensor<3xf64>
    %dcv = stablehlo.divide %num_c, %den : tensor<3xf64>
    %z = stablehlo.constant dense<0.0> : tensor<f64>
    %dc = stablehlo.reduce(%dcv init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    check.expect_almost_eq %marked#2, %dc, tolerance = 1.0e-8 : tensor<f64>
    return
  }

  func.func private @fixed_point(%u0: tensor<3xf64>, %p: tensor<3xf64>, %cb: tensor<3xf64>) -> tensor<3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %cap = stablehlo.constant dense<500> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %u = %u0) : tensor<i64>, tensor<3xf64>
     cond {
      %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    } do {
      %s = stablehlo.sine %u : tensor<3xf64>
      %cs = stablehlo.multiply %cb, %s : tensor<3xf64>
      %p2 = stablehlo.multiply %p, %p : tensor<3xf64>
      %un = stablehlo.add %cs, %p2 : tensor<3xf64>
      %in = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %in, %un : tensor<i64>, tensor<3xf64>
    }
    return %r#1 : tensor<3xf64>
  }
}
