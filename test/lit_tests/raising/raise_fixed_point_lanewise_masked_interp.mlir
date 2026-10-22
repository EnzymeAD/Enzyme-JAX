// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --arith-raise="stablehlo=true" | stablehlo-translate - --interpret --allow-unregistered-dialect

// A per-thread secant iteration for sqrt(a), marked as a fixed point. Raised,
// the loop runs over the whole grid until the slowest lane has converged, and
// each lane must stop stepping once its own test fails, exactly as the threads
// would: one more secant step on a lane that has converged exactly is
// (x1 - x0) / (f1 - f0) = 0/0. Lane 0 has converged before the first step,
// lanes 1-3 take a handful of steps and lane 4 takes dozens. The second kernel
// puts the loop under a lane-wise if.

module {
  func.func private @secant(%x: memref<5xf64, 1>, %a: memref<5xf64, 1>) {
    %one = arith.constant 1.0 : f64
    %half = arith.constant 0.5 : f64
    %tol = arith.constant 1.0e-14 : f64
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %cap = arith.constant 500 : i64
    affine.parallel (%i) = (0) to (5) {
      %ai = affine.load %a[%i] : memref<5xf64, 1>
      %ap1 = arith.addf %ai, %one : f64
      %x1i = arith.mulf %ap1, %half : f64
      %r:3 = scf.while (%x0 = %ai, %x1 = %x1i, %k = %c0) : (f64, f64, i64) -> (f64, f64, i64) {
        %d = arith.subf %x1, %x0 : f64
        %ad = math.absf %d : f64
        %ax1 = math.absf %x1 : f64
        %t = arith.mulf %ax1, %tol : f64
        %moving = arith.cmpf ogt, %ad, %t : f64
        %capped = arith.cmpi slt, %k, %cap : i64
        %go = arith.andi %moving, %capped : i1
        scf.condition(%go) %x0, %x1, %k : f64, f64, i64
      } do {
      ^bb0(%x0: f64, %x1: f64, %k: i64):
        %x0sq = arith.mulf %x0, %x0 : f64
        %f0 = arith.subf %x0sq, %ai : f64
        %x1sq = arith.mulf %x1, %x1 : f64
        %f1 = arith.subf %x1sq, %ai : f64
        %dx = arith.subf %x1, %x0 : f64
        %df = arith.subf %f1, %f0 : f64
        %num = arith.mulf %f1, %dx : f64
        %q = arith.divf %num, %df : f64
        %x2 = arith.subf %x1, %q : f64
        %kn = arith.addi %k, %c1 : i64
        scf.yield %x1, %x2, %kn : f64, f64, i64
      } attributes {enzyme.fixed_point}
      affine.store %r#1, %x[%i] : memref<5xf64, 1>
    }
    return
  }

  func.func private @secant_guarded(%x: memref<6xf64, 1>, %iters: memref<6xi64, 1>, %a: memref<6xf64, 1>) {
    %one = arith.constant 1.0 : f64
    %half = arith.constant 0.5 : f64
    %tol = arith.constant 1.0e-14 : f64
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %cap = arith.constant 500 : i64
    affine.parallel (%i) = (0) to (6) {
     affine.if affine_set<(d0) : (4 - d0 >= 0)>(%i) {
      %ai = affine.load %a[%i] : memref<6xf64, 1>
      %ap1 = arith.addf %ai, %one : f64
      %x1i = arith.mulf %ap1, %half : f64
      %r:3 = scf.while (%x0 = %ai, %x1 = %x1i, %k = %c0) : (f64, f64, i64) -> (f64, f64, i64) {
        %d = arith.subf %x1, %x0 : f64
        %ad = math.absf %d : f64
        %ax1 = math.absf %x1 : f64
        %t = arith.mulf %ax1, %tol : f64
        %moving = arith.cmpf ogt, %ad, %t : f64
        %capped = arith.cmpi slt, %k, %cap : i64
        %go = arith.andi %moving, %capped : i1
        scf.condition(%go) %x0, %x1, %k : f64, f64, i64
      } do {
      ^bb0(%x0: f64, %x1: f64, %k: i64):
        %x0sq = arith.mulf %x0, %x0 : f64
        %f0 = arith.subf %x0sq, %ai : f64
        %x1sq = arith.mulf %x1, %x1 : f64
        %f1 = arith.subf %x1sq, %ai : f64
        %dx = arith.subf %x1, %x0 : f64
        %df = arith.subf %f1, %f0 : f64
        %num = arith.mulf %f1, %dx : f64
        %q = arith.divf %num, %df : f64
        %x2 = arith.subf %x1, %q : f64
        %kn = arith.addi %k, %c1 : i64
        scf.yield %x1, %x2, %kn : f64, f64, i64
      } attributes {enzyme.fixed_point}
      affine.store %r#1, %x[%i] : memref<6xf64, 1>
      affine.store %r#2, %iters[%i] : memref<6xi64, 1>
     }
    }
    return
  }

  func.func @main() {
    %x = stablehlo.constant dense<0.0> : tensor<5xf64>
    %a = stablehlo.constant dense<[1.0, 4.0, 2.0, 9.0, 1.0e12]> : tensor<5xf64>
    %r = enzymexla.jit_call @secant (%x, %a) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<5xf64>, tensor<5xf64>) -> tensor<5xf64>
    check.expect_almost_eq_const %r, dense<[1.0, 2.0, 1.4142135623730951, 3.0, 1.0e6]> : tensor<5xf64>

    // The same loop under a bounds check, as a KernelAbstractions kernel has
    // it: lane 5 is outside and never enters the loop (its a = -1 would not
    // converge). Each lane stops after its own iteration count.
    %gx = stablehlo.constant dense<0.0> : tensor<6xf64>
    %gk = stablehlo.constant dense<0> : tensor<6xi64>
    %ga = stablehlo.constant dense<[1.0, 4.0, 2.0, 9.0, 1.0e12, -1.0]> : tensor<6xf64>
    %g:2 = enzymexla.jit_call @secant_guarded (%gx, %gk, %ga) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<6xf64>, tensor<6xi64>, tensor<6xf64>) -> (tensor<6xf64>, tensor<6xi64>)
    check.expect_almost_eq_const %g#0, dense<[1.0, 2.0, 1.4142135623730951, 3.0, 1.0e6, 0.0]> : tensor<6xf64>
    check.expect_eq_const %g#1, dense<[0, 7, 6, 8, 35, 0]> : tensor<6xi64>
    return
  }
}
