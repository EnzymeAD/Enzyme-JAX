// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

// The reverse of a stablehlo.while is given its own counter, initialised to
// numIters-1 and decremented each step, so it already holds the forward index of
// the iteration being reversed. Taping the forward induction variable therefore
// materialises an identity map `tape[i] = i`.
//
// Beyond the wasted buffer, reading it back makes every reverse index
// data-dependent, which blinds bounds analysis and stops while_induction_reduction
// (and while_dus / while_dus_ds_simplify) from shrinking the loop-carried buffers.
//
// Enzyme's shared loop handling seeds `fwdrevmap` with the forward IV before running
// the min cut so it is never a cut root (RemovalUtils.h, used by scf/affine).
// stablehlo.while builds its reverse counter after the min cut has run, so it must
// drop the induction-variable cache explicitly.

module {
  func.func private @loop(%x: tensor<8x4xf32>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %init = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
    %r:2 = stablehlo.while(%i = %c0, %acc = %init) : tensor<i64>, tensor<4xf32>
    cond {
      %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 4] : (tensor<8x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
      %rowr = stablehlo.reshape %row : (tensor<1x4xf32>) -> tensor<4xf32>
      %prod = stablehlo.multiply %acc, %rowr : tensor<4xf32>
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %prod : tensor<i64>, tensor<4xf32>
    }
    return %r#1 : tensor<4xf32>
  }
  func.func @main(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
    %0 = enzyme.autodiff @loop(%x, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<8x4xf32>, tensor<4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

// The induction variable must not be taped: no i64 buffer is carried, and the
// augmented forward loop carries only the counter, the primal accumulator and the
// one genuine cache.
// CHECK-LABEL:   func.func private @diffeloop(
// CHECK-NOT:       tensor<8xi64>
// CHECK:           stablehlo.while(%iterArg = %{{.*}}, %iterArg_{{.*}} = %{{.*}}, %iterArg_{{.*}} = %{{.*}}) : tensor<i64>, tensor<4xf32>, tensor<8x4xf32>
// CHECK-NOT:       tensor<8xi64>
// CHECK:           return
