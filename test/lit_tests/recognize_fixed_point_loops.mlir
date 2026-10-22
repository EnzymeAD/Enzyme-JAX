// RUN: enzymexlamlir-opt %s --recognize-fixed-point-loops --split-input-file | FileCheck %s

// A kernel's Newton loop as the CFG lifting leaves it: a rotated scf.while
// in LLVM-dialect arithmetic, testing the step |q| > tol with a counter cap.
// Recognized: |q| is derived from the subtraction x^2 - a of the state.
llvm.func @newton_kernel(%n: i32, %x0: f64, %a: f64, %tol: f64) -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2.0 : f64) : f64
  %r:2 = scf.while (%i = %c0, %x = %x0) : (i32, f64) -> (i32, f64) {
    %x2 = llvm.fmul %x, %x : f64
    %res = llvm.fsub %x2, %a : f64
    %d = llvm.fmul %two, %x : f64
    %q = llvm.fdiv %res, %d : f64
    %xn = llvm.fsub %x, %q : f64
    %aq = llvm.intr.fabs(%q) : (f64) -> f64
    %gt = llvm.fcmp "ogt" %aq, %tol : f64
    %in = llvm.add %i, %c1 : i32
    %lt = llvm.icmp "slt" %in, %n : i32
    %go = llvm.and %gt, %lt : i1
    scf.condition(%go) %in, %xn : i32, f64
  } do {
  ^bb0(%i2: i32, %x2: f64):
    scf.yield %i2, %x2 : i32, f64
  }
  llvm.return %r#1 : f64
}
// CHECK-LABEL: @newton_kernel
// CHECK: scf.while
// CHECK: } attributes {enzyme.fixed_point}

// -----

// A traced Picard iteration: the residual max|u_new - u| is carried and
// tested in the condition. Recognized.
func.func @picard(%u0: tensor<3xf64>, %p: tensor<3xf64>, %c: tensor<3xf64>) -> tensor<3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %cap = stablehlo.constant dense<500> : tensor<i64>
  %tol = stablehlo.constant dense<1.0e-12> : tensor<f64>
  %one = stablehlo.constant dense<1.0> : tensor<f64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %r:3 = stablehlo.while(%i = %c0, %u = %u0, %err = %one) : tensor<i64>, tensor<3xf64>, tensor<f64>
   cond {
    %lt = stablehlo.compare LT, %i, %cap : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %gt = stablehlo.compare GT, %err, %tol : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %go = stablehlo.and %lt, %gt : tensor<i1>
    stablehlo.return %go : tensor<i1>
  } do {
    %s = stablehlo.sine %u : tensor<3xf64>
    %cs = stablehlo.multiply %c, %s : tensor<3xf64>
    %un = stablehlo.add %cs, %p : tensor<3xf64>
    %d = stablehlo.subtract %un, %u : tensor<3xf64>
    %ad = stablehlo.abs %d : tensor<3xf64>
    %e = stablehlo.reduce(%ad init: %zero) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %in = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in, %un, %e : tensor<i64>, tensor<3xf64>, tensor<f64>
  }
  return %r#1 : tensor<3xf64>
}
// CHECK-LABEL: @picard
// CHECK: stablehlo.while{{.*}} attributes {enzyme.fixed_point}

// -----

// A time integration stopping at t_end. The condition compares a carried
// value against an invariant, but t is an accumulator (t + dt with dt
// independent of t) and nothing in it is a change of state. Not recognized.
func.func @time_loop(%u0: tensor<3xf64>, %dt: tensor<f64>, %tend: tensor<f64>) -> tensor<3xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %r:2 = stablehlo.while(%t = %zero, %u = %u0) : tensor<f64>, tensor<3xf64>
   cond {
    %go = stablehlo.compare LT, %t, %tend : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %go : tensor<i1>
  } do {
    %s = stablehlo.sine %u : tensor<3xf64>
    %dtb = stablehlo.broadcast_in_dim %dt, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %inc = stablehlo.multiply %dtb, %s : tensor<3xf64>
    %un = stablehlo.add %u, %inc : tensor<3xf64>
    %tn = stablehlo.add %t, %dt : tensor<f64>
    stablehlo.return %tn, %un : tensor<f64>, tensor<3xf64>
  }
  return %r#1 : tensor<3xf64>
}
// CHECK-LABEL: @time_loop
// CHECK-NOT: enzyme.fixed_point

// -----

// The same time loop written with a remaining-time test, t_end - t > 0: the
// subtraction looks like a change of state, but t is still an accumulator.
// Not recognized.
func.func @time_loop_remaining(%u0: tensor<3xf64>, %dt: tensor<f64>, %tend: tensor<f64>) -> tensor<3xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %r:2 = stablehlo.while(%t = %zero, %u = %u0) : tensor<f64>, tensor<3xf64>
   cond {
    %rem = stablehlo.subtract %tend, %t : tensor<f64>
    %go = stablehlo.compare GT, %rem, %zero : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %go : tensor<i1>
  } do {
    %s = stablehlo.sine %u : tensor<3xf64>
    %dtb = stablehlo.broadcast_in_dim %dt, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %inc = stablehlo.multiply %dtb, %s : tensor<3xf64>
    %un = stablehlo.add %u, %inc : tensor<3xf64>
    %tn = stablehlo.add %t, %dt : tensor<f64>
    stablehlo.return %tn, %un : tensor<f64>, tensor<3xf64>
  }
  return %r#1 : tensor<3xf64>
}
// CHECK-LABEL: @time_loop_remaining
// CHECK-NOT: enzyme.fixed_point

// -----

// Conjugate gradient: converges on the residual norm, but x is updated as
// x + alpha p where alpha p does not depend on x. Not recognized.
func.func @cg(%x0: tensor<3xf64>, %r0: tensor<3xf64>, %A: tensor<3x3xf64>, %tol: tensor<f64>) -> tensor<3xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %r:3 = stablehlo.while(%x = %x0, %r = %r0, %p = %r0) : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
   cond {
    %rr = stablehlo.multiply %r, %r : tensor<3xf64>
    %n = stablehlo.reduce(%rr init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %go = stablehlo.compare GT, %n, %tol : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %go : tensor<i1>
  } do {
    %Ap = stablehlo.dot_general %A, %p, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %rr = stablehlo.multiply %r, %r : tensor<3xf64>
    %num = stablehlo.reduce(%rr init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %pAp = stablehlo.multiply %p, %Ap : tensor<3xf64>
    %den = stablehlo.reduce(%pAp init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %alpha = stablehlo.divide %num, %den : tensor<f64>
    %ab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %ap = stablehlo.multiply %ab, %p : tensor<3xf64>
    %xn = stablehlo.add %x, %ap : tensor<3xf64>
    %aAp = stablehlo.multiply %ab, %Ap : tensor<3xf64>
    %rn = stablehlo.subtract %r, %aAp : tensor<3xf64>
    %rnrn = stablehlo.multiply %rn, %rn : tensor<3xf64>
    %nn = stablehlo.reduce(%rnrn init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %beta = stablehlo.divide %nn, %num : tensor<f64>
    %bb = stablehlo.broadcast_in_dim %beta, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %bp = stablehlo.multiply %bb, %p : tensor<3xf64>
    %pn = stablehlo.add %rn, %bp : tensor<3xf64>
    stablehlo.return %xn, %rn, %pn : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }
  return %r#0 : tensor<3xf64>
}
// CHECK-LABEL: @cg
// CHECK-NOT: enzyme.fixed_point

// -----

// An iteration whose step depends on the counter (a Chebyshev-like weight).
// Converges on the change of state, but the map changes every iteration.
// Not recognized.
func.func @counter_dependent(%u0: tensor<3xf64>, %p: tensor<3xf64>, %tol: tensor<f64>) -> tensor<3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %one = stablehlo.constant dense<1.0> : tensor<f64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %r:3 = stablehlo.while(%i = %c0, %u = %u0, %err = %one) : tensor<i64>, tensor<3xf64>, tensor<f64>
   cond {
    %go = stablehlo.compare GT, %err, %tol : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %go : tensor<i1>
  } do {
    %fi = stablehlo.convert %i : (tensor<i64>) -> tensor<f64>
    %w = stablehlo.divide %one, %fi : tensor<f64>
    %wb = stablehlo.broadcast_in_dim %w, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %s = stablehlo.sine %u : tensor<3xf64>
    %ws = stablehlo.multiply %wb, %s : tensor<3xf64>
    %un = stablehlo.add %ws, %p : tensor<3xf64>
    %d = stablehlo.subtract %un, %u : tensor<3xf64>
    %ad = stablehlo.abs %d : tensor<3xf64>
    %e = stablehlo.reduce(%ad init: %zero) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %in = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in, %un, %e : tensor<i64>, tensor<3xf64>, tensor<f64>
  }
  return %r#1 : tensor<3xf64>
}
// CHECK-LABEL: @counter_dependent
// CHECK-NOT: enzyme.fixed_point

// -----

// An explicit opt-out is respected.
func.func @opted_out(%u0: tensor<3xf64>, %p: tensor<3xf64>, %tol: tensor<f64>) -> tensor<3xf64> {
  %one = stablehlo.constant dense<1.0> : tensor<f64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %r:2 = stablehlo.while(%u = %u0, %err = %one) : tensor<3xf64>, tensor<f64> attributes {enzyme.fixed_point = false}
   cond {
    %go = stablehlo.compare GT, %err, %tol : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %go : tensor<i1>
  } do {
    %s = stablehlo.sine %u : tensor<3xf64>
    %un = stablehlo.add %s, %p : tensor<3xf64>
    %d = stablehlo.subtract %un, %u : tensor<3xf64>
    %ad = stablehlo.abs %d : tensor<3xf64>
    %e = stablehlo.reduce(%ad init: %zero) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    stablehlo.return %un, %e : tensor<3xf64>, tensor<f64>
  }
  return %r#0 : tensor<3xf64>
}
// CHECK-LABEL: @opted_out
// CHECK: attributes {enzyme.fixed_point = false}

// -----

// Breeze's NewtonSolver temperature inversion as the CFG lifting leaves it:
// the cap test `iter >= maxiter` is an scf.if whose then-branch is the exit
// (yielding a false predicate), the step and the test `|ΔT| > max(reltol T,
// abstol)` live in the else-branch, the predicate passes through extui/trunci,
// and the before region forwards three values from two carries. Recognized.
func.func @breeze_newton(%T0: f64, %theta: f64, %rho: f64) -> f64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c20 = arith.constant 20 : i64
  %false = arith.constant 0 : i32
  %zero = arith.constant 0.0 : f64
  %abstol = arith.constant 1.0e-10 : f64
  %reltol = arith.constant 0.0 : f64
  %kappa = arith.constant 0.2857 : f64
  %one = arith.constant 1.0 : f64
  %poison = ub.poison : f64
  %poisoni = ub.poison : i64
  %r:3 = scf.while (%T = %T0, %iter = %c0) : (f64, i64) -> (f64, i64, f64) {
    %capped = arith.cmpi sge, %iter, %c20 : i64
    %s:4 = scf.if %capped -> (f64, i64, f64, i32) {
      scf.yield %poison, %poisoni, %T, %false : f64, i64, f64, i32
    } else {
      %rt = arith.mulf %rho, %T : f64
      %phi = math.powf %rt, %kappa : f64
      %phith = arith.mulf %theta, %phi : f64
      %res = arith.subf %T, %phith : f64
      %nres = arith.negf %res : f64
      %kp = arith.mulf %phith, %kappa : f64
      %kpt = arith.divf %kp, %T : f64
      %der = arith.subf %one, %kpt : f64
      %dT = arith.divf %nres, %der : f64
      %Tn = arith.addf %T, %dT : f64
      %in = arith.addi %iter, %c1 : i64
      %adT = math.absf %dT : f64
      %rel = arith.mulf %Tn, %reltol : f64
      %tol = arith.maximumf %rel, %abstol : f64
      %go = arith.cmpf olt, %tol, %adT : f64
      %goi = arith.extui %go : i1 to i32
      scf.yield %Tn, %in, %Tn, %goi : f64, i64, f64, i32
    }
    %go1 = arith.trunci %s#3 : i32 to i1
    scf.condition(%go1) %s#0, %s#1, %s#2 : f64, i64, f64
  } do {
  ^bb0(%T: f64, %iter: i64, %Tout: f64):
    scf.yield %T, %iter : f64, i64
  }
  return %r#2 : f64
}
// CHECK-LABEL: @breeze_newton
// CHECK: scf.while
// CHECK: } attributes {enzyme.fixed_point}
