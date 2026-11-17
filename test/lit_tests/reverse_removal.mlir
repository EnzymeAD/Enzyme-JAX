// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reverse_removal},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  // Issue #1237: push reverse upward through multi-use elementwise DAG,
  // broadcast_in_dim, enzymexla.wrap, and padded reduce_window to cancel
  // leaf reverses and invariant broadcasts/splats.
  func.func @reverse_removal_issue1237(
      %arg0: tensor<3x8x10xf64>,
      %arg1: tensor<3x8x10xf64>,
      %arg2: tensor<3xf64>,
      %arg3: tensor<3xf64>,
      %arg4: tensor<8x12xf64>) -> tensor<3x8x12xf64> {
    %cst0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst2 = stablehlo.constant dense<2.000000e+00> : tensor<3x8x12xf64>
    %cst3 = stablehlo.constant dense<3.000000e+00> : tensor<3xf64>

    %w0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<3x8x10xf64>) -> tensor<3x8x12xf64>
    %r0 = stablehlo.reverse %w0, dims = [0] : tensor<3x8x12xf64>
    %w1 = "enzymexla.wrap"(%arg1) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<3x8x10xf64>) -> tensor<3x8x12xf64>
    %r1 = stablehlo.reverse %w1, dims = [0] : tensor<3x8x12xf64>
    %r2 = stablehlo.reverse %arg2, dims = [0] : tensor<3xf64>

    %m0 = stablehlo.multiply %r0, %cst2 : tensor<3x8x12xf64>
    %s0 = stablehlo.sqrt %r1 : tensor<3x8x12xf64>
    %neg = stablehlo.negate %r2 : tensor<3xf64>
    %scaled = stablehlo.multiply %neg, %cst3 : tensor<3xf64>
    %b0 = stablehlo.broadcast_in_dim %scaled, dims = [0] : (tensor<3xf64>) -> tensor<3x8x12xf64>

    // Multi-use interior values inside the cone (%m0, %s0, %b0)
    %p0 = stablehlo.add %m0, %s0 : tensor<3x8x12xf64>
    %p1 = stablehlo.multiply %p0, %m0 : tensor<3x8x12xf64>
    %p2 = stablehlo.add %p1, %s0 : tensor<3x8x12xf64>
    %p3 = stablehlo.multiply %p2, %b0 : tensor<3x8x12xf64>

    %r3 = stablehlo.reverse %arg3, dims = [0] : tensor<3xf64>
    %b1 = stablehlo.broadcast_in_dim %r3, dims = [0] : (tensor<3xf64>) -> tensor<3x8x12xf64>
    %prod = stablehlo.multiply %b1, %p3 : tensor<3x8x12xf64>

    // Reversible invariant broadcast along dims [1, 2]
    %b_inv = stablehlo.broadcast_in_dim %arg4, dims = [1, 2] : (tensor<8x12xf64>) -> tensor<3x8x12xf64>

    // Padded reduce_window along dim 0 flips padding [2, 0] -> [0, 2]
    %rw = "stablehlo.reduce_window"(%prod, %cst0) <{
      base_dilations = array<i64: 1, 1, 1>,
      padding = dense<[[2, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
      window_dilations = array<i64: 1, 1, 1>,
      window_dimensions = array<i64: 3, 1, 1>,
      window_strides = array<i64: 1, 1, 1>
    }> ({
    ^bb0(%lhs: tensor<f64>, %rhs: tensor<f64>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f64>
      stablehlo.return %sum : tensor<f64>
    }) : (tensor<3x8x12xf64>, tensor<f64>) -> tensor<3x8x12xf64>

    %diff = stablehlo.subtract %b_inv, %rw : tensor<3x8x12xf64>
    %out = stablehlo.reverse %diff, dims = [0] : tensor<3x8x12xf64>
    return %out : tensor<3x8x12xf64>
  }

  // Negative test: an unreversed non-invariant leaf prevents pushing reverse.
  func.func @do_not_push_into_unreversed_leaf(
      %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %r0 = stablehlo.reverse %arg0, dims = [0] : tensor<4x8xf32>
    %add = stablehlo.add %r0, %arg1 : tensor<4x8xf32>
    %out = stablehlo.reverse %add, dims = [0] : tensor<4x8xf32>
    return %out : tensor<4x8xf32>
  }

  // Negative test: an external user of an interior op prevents duplicating ops.
  func.func @do_not_duplicate_escaped_interior(
      %arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4x8xf32>
    %r0 = stablehlo.reverse %arg0, dims = [0] : tensor<4x8xf32>
    %add = stablehlo.add %r0, %cst : tensor<4x8xf32>
    %out = stablehlo.reverse %add, dims = [0] : tensor<4x8xf32>
    return %out, %add : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// CHECK-LABEL: func.func @reverse_removal_issue1237(
// CHECK-NOT: stablehlo.reverse
// CHECK: "stablehlo.reduce_window"
// CHECK-SAME: padding = dense<{{\[\[}}0, 2], [0, 0], [0, 0]]> : tensor<3x2xi64>
// CHECK-NOT: stablehlo.reverse
// CHECK: return

// CHECK-LABEL: func.func @do_not_push_into_unreversed_leaf(
// CHECK: %[[R0:.+]] = stablehlo.reverse %arg0, dims = [0]
// CHECK: %[[ADD:.+]] = stablehlo.add %[[R0]], %arg1
// CHECK: %[[OUT:.+]] = stablehlo.reverse %[[ADD]], dims = [0]
// CHECK: return %[[OUT]]

// CHECK-LABEL: func.func @do_not_duplicate_escaped_interior(
// CHECK: %[[ER0:.+]] = stablehlo.reverse %arg0, dims = [0]
// CHECK: %[[EADD:.+]] = stablehlo.add %[[ER0]], %cst
// CHECK: %[[EOUT:.+]] = stablehlo.reverse %[[EADD]], dims = [0]
// CHECK: return %[[EOUT]], %[[EADD]]
