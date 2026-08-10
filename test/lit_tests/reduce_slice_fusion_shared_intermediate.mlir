// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// `state_i = state_{i-1} + x[i]` where every `state_i` is also read by its own
// `multiply`. Fusing the chain into a reduce would rewrite each `state_i` into a
// reduce over a growing prefix; because the other users keep the chain alive the
// prefixes are recomputed rather than shared, turning O(N) work into O(N^2).
// The chain must be left alone.

func.func @shared_intermediate(%x: tensor<4xf32>) -> tensor<f32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %s0 = stablehlo.slice %x [0:1] : (tensor<4xf32>) -> tensor<1xf32>
  %e0 = stablehlo.reshape %s0 : (tensor<1xf32>) -> tensor<f32>
  %st1 = stablehlo.add %zero, %e0 : tensor<f32>
  %q1 = stablehlo.multiply %st1, %st1 : tensor<f32>
  %s1 = stablehlo.slice %x [1:2] : (tensor<4xf32>) -> tensor<1xf32>
  %e1 = stablehlo.reshape %s1 : (tensor<1xf32>) -> tensor<f32>
  %st2 = stablehlo.add %st1, %e1 : tensor<f32>
  %q2 = stablehlo.multiply %st2, %st2 : tensor<f32>
  %s2 = stablehlo.slice %x [2:3] : (tensor<4xf32>) -> tensor<1xf32>
  %e2 = stablehlo.reshape %s2 : (tensor<1xf32>) -> tensor<f32>
  %st3 = stablehlo.add %st2, %e2 : tensor<f32>
  %q3 = stablehlo.multiply %st3, %st3 : tensor<f32>
  %s3 = stablehlo.slice %x [3:4] : (tensor<4xf32>) -> tensor<1xf32>
  %e3 = stablehlo.reshape %s3 : (tensor<1xf32>) -> tensor<f32>
  %st4 = stablehlo.add %st3, %e3 : tensor<f32>
  %q4 = stablehlo.multiply %st4, %st4 : tensor<f32>
  %l1 = stablehlo.add %q1, %q2 : tensor<f32>
  %l2 = stablehlo.add %l1, %q3 : tensor<f32>
  %l3 = stablehlo.add %l2, %q4 : tensor<f32>
  return %l3 : tensor<f32>
}

// The accumulator chain survives: each step adds one element to the previous
// state (%8 uses %4, %12 uses %8) rather than re-reducing a growing prefix, so
// the op count stays linear in the chain length. Fusing the first two elements
// is fine -- it reads two slices and reduces no existing accumulator.

// CHECK:  func.func @shared_intermediate(%arg0: tensor<4xf32>) -> tensor<f32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %1 : tensor<f32>
// CHECK-NEXT:    %3 = stablehlo.slice %arg0 [0:2] : (tensor<4xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %4 : tensor<f32>
// CHECK-NEXT:    %6 = stablehlo.slice %arg0 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %8 = stablehlo.add %4, %7 : tensor<f32>
// CHECK-NEXT:    %9 = stablehlo.multiply %8, %8 : tensor<f32>
// CHECK-NEXT:    %10 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %12 = stablehlo.add %8, %11 : tensor<f32>
// CHECK-NEXT:    %13 = stablehlo.multiply %12, %12 : tensor<f32>
// CHECK-NEXT:    %14 = stablehlo.add %2, %5 : tensor<f32>
// CHECK-NEXT:    %15 = stablehlo.add %14, %9 : tensor<f32>
// CHECK-NEXT:    %16 = stablehlo.add %15, %13 : tensor<f32>
// CHECK-NEXT:    return %16 : tensor<f32>
// CHECK-NEXT:  }
