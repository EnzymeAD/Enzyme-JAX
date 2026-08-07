// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

// CONSTANT_CHECKPOINTING with a period that does NOT divide the trip count
// (10 iterations, period 3 -> nOuter=4, last outer block has actualInner=1).
//
// Before the fix, the reverse-pass cache buffer ended up typed
// `tensor<?x3xf64>` because:
//   - the inner re-forward loop's limit is a runtime `min(3, remainingIters)`,
//   - `WhileLoopInfo::isConstant()` returns false for that limit,
//   - so cache rewriting fell back to `tensor.empty + dynamic_pad -> ?x...`.
// XLA cannot translate dynamic_pad, and the stablehlo interpreter rejects the
// dynamic result type too. This test cached a *tensor* (not a scalar) to
// exercise that path.
//
// The fix annotates the MinOp with `enzymexla.bounds = [[1, nInner]]` and
// teaches the cache rewriter to use that bound to size the buffer statically.
// We verify both the IR shape and end-to-end correctness via the interpreter.

// CHECK: func.func private @diffe
// The boundary-block path produces a `stablehlo.min` whose result carries the
// static upper bound annotation.
// CHECK: stablehlo.min{{.*}}enzymexla.bounds
// The inner forward cache buffer is statically shaped tensor<3x3xf64>, not
// tensor<?x3xf64>. (Outer dim 3 = nInner, inner dim 3 = element shape.)
// CHECK-NOT: tensor<?x3xf64>
// CHECK: tensor<3x3xf64>
// No dynamic_pad in the cache construction.
// CHECK-NOT: stablehlo.dynamic_pad

module {
  // Reference: no checkpointing.
  func.func @ref(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_n = stablehlo.constant dense<10> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_x = %arg0) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut}
     cond {
      %1 = stablehlo.compare LT, %iterArg, %c_n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.sine %iterArg_x : tensor<3xf64>
      stablehlo.return %1, %2 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  // Checkpointed: 10 iters, period 3 -> nOuter=4, boundary block runs 1 iter.
  func.func @ckpt(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_n = stablehlo.constant dense<10> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_x = %arg0) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true, enzymexla.checkpoint_period = 3 : i64}
     cond {
      %1 = stablehlo.compare LT, %iterArg, %c_n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.sine %iterArg_x : tensor<3xf64>
      stablehlo.return %1, %2 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @main() {
    %input = stablehlo.constant dense<[0.3, 0.6, 0.9]> : tensor<3xf64>
    %diffe = stablehlo.constant dense<1.0> : tensor<3xf64>

    %ref_g:2 = enzyme.autodiff @ref(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)

    %ckpt_g:2 = enzyme.autodiff @ckpt(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)

    // Gradient with boundary-block checkpointing must match no-checkpointing.
    check.expect_almost_eq %ckpt_g#0, %ref_g#0 : tensor<3xf64>
    check.expect_almost_eq %ckpt_g#1, %ref_g#1 : tensor<3xf64>

    return
  }
}
