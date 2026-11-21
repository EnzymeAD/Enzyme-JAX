// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // 1. #1836 (linear chain): y + x + x + ... + x (9 times) -> x * 9.0 + y.
  // CHECK-LABEL: func.func private @self_add_linear_chain
  // CHECK: %[[C9:.*]] = stablehlo.constant dense<9.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %arg0, %[[C9]] : tensor<2x2xf32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[B]] : tensor<2x2xf32>
  // CHECK-NOT: stablehlo.add
  // CHECK: return %[[ADD]] : tensor<2x2xf32>
  func.func private @self_add_linear_chain(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
    %b = stablehlo.broadcast_in_dim %sum, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %a1 = stablehlo.add %b, %x : tensor<2x2xf32>
    %a2 = stablehlo.add %a1, %x : tensor<2x2xf32>
    %a3 = stablehlo.add %a2, %x : tensor<2x2xf32>
    %a4 = stablehlo.add %a3, %x : tensor<2x2xf32>
    %a5 = stablehlo.add %a4, %x : tensor<2x2xf32>
    %a6 = stablehlo.add %a5, %x : tensor<2x2xf32>
    %a7 = stablehlo.add %a6, %x : tensor<2x2xf32>
    %a8 = stablehlo.add %a7, %x : tensor<2x2xf32>
    %a9 = stablehlo.add %a8, %x : tensor<2x2xf32>
    return %a9 : tensor<2x2xf32>
  }

  // 2. #1836 (doubling DAG): (b + x) + (((x + x) + (x + x)) + ...) -> b + x * 9.0.
  // CHECK-LABEL: func.func private @self_add_doubling_dag
  // CHECK: %[[C9D:.*]] = stablehlo.constant dense<9.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[BD:.*]] = stablehlo.broadcast_in_dim
  // CHECK: %[[MULD:.*]] = stablehlo.multiply %arg0, %[[C9D]] : tensor<2x2xf32>
  // CHECK: %[[ADDD:.*]] = stablehlo.add %[[BD]], %[[MULD]] : tensor<2x2xf32>
  // CHECK-NOT: stablehlo.add
  // CHECK: return %[[ADDD]] : tensor<2x2xf32>
  func.func private @self_add_doubling_dag(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%x init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
    %b = stablehlo.broadcast_in_dim %sum, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %d2 = stablehlo.add %b, %x : tensor<2x2xf32>
    %d3 = stablehlo.add %x, %x : tensor<2x2xf32>
    %d4 = stablehlo.add %d3, %d3 : tensor<2x2xf32>
    %d5 = stablehlo.add %d4, %d4 : tensor<2x2xf32>
    %d6 = stablehlo.add %d2, %d5 : tensor<2x2xf32>
    return %d6 : tensor<2x2xf32>
  }

  // 3. #1867: reshape collapsing non-contracting dims around dot_general folds away.
  // CHECK-LABEL: func.func private @dot_reshape_rhs_collapse
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0]{{.*}} : (tensor<3x3xf32>, tensor<3x2x4xf32>) -> tensor<3x2x4xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<3x2x4xf32>
  func.func private @dot_reshape_rhs_collapse(%arg0: tensor<3x2x4xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x2x4xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<3x2x4xf32>) -> tensor<3x8xf32>
    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf32>, tensor<3x8xf32>) -> tensor<3x8xf32>
    %2 = stablehlo.reshape %1 : (tensor<3x8xf32>) -> tensor<3x2x4xf32>
    return %2 : tensor<3x2x4xf32>
  }

  // 4. #1867: free-dim reshapes on both sides of a middle contracting dim + batching dim.
  // CHECK-LABEL: func.func private @dot_reshape_batched_split_blocks
  // CHECK-NEXT: %[[DOTB:.*]] = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [3] x [1]{{.*}} : (tensor<2x2x3x2x2xf32>, tensor<2x2x3xf32>) -> tensor<2x2x3x2x3xf32>
  // CHECK-NEXT: return %[[DOTB]] : tensor<2x2x3x2x3xf32>
  func.func private @dot_reshape_batched_split_blocks(%lhs: tensor<2x2x3x2x2xf32>, %rhs: tensor<2x2x3xf32>) -> tensor<2x2x3x2x3xf32> {
    %0 = stablehlo.reshape %lhs : (tensor<2x2x3x2x2xf32>) -> tensor<2x6x2x2xf32>
    %1 = stablehlo.dot_general %0, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x6x2x2xf32>, tensor<2x2x3xf32>) -> tensor<2x6x2x3xf32>
    %2 = stablehlo.reshape %1 : (tensor<2x6x2x3xf32>) -> tensor<2x2x3x2x3xf32>
    return %2 : tensor<2x2x3x2x3xf32>
  }

  func.func @main() {
    %m = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %c1 = func.call @self_add_linear_chain(%m) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    check.expect_almost_eq_const %c1, dense<[[19.0, 28.0], [37.0, 46.0]]> : tensor<2x2xf32>

    %c2 = func.call @self_add_doubling_dag(%m) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    check.expect_almost_eq_const %c2, dense<[[19.0, 28.0], [37.0, 46.0]]> : tensor<2x2xf32>

    %a0 = stablehlo.constant dense<[
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
      [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
      [[2.0, 1.0, 0.0, -1.0], [-2.0, 0.0, 1.0, 2.0]]
    ]> : tensor<3x2x4xf32>
    %a1 = stablehlo.constant dense<[
      [1.0, 2.0, 0.0],
      [0.0, 1.0, 1.0],
      [2.0, 0.0, -1.0]
    ]> : tensor<3x3xf32>
    %dr1 = func.call @dot_reshape_rhs_collapse(%a0, %a1) : (tensor<3x2x4xf32>, tensor<3x3xf32>) -> tensor<3x2x4xf32>
    %exp1 = stablehlo.dot_general %a1, %a0, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x2x4xf32>) -> tensor<3x2x4xf32>
    check.expect_almost_eq %dr1, %exp1 : tensor<3x2x4xf32>

    %lhs_iota = stablehlo.iota dim = 0 : tensor<48xf32>
    %lhs = stablehlo.reshape %lhs_iota : (tensor<48xf32>) -> tensor<2x2x3x2x2xf32>
    %rhs_iota = stablehlo.iota dim = 0 : tensor<12xf32>
    %rhs = stablehlo.reshape %rhs_iota : (tensor<12xf32>) -> tensor<2x2x3xf32>
    %dr2 = func.call @dot_reshape_batched_split_blocks(%lhs, %rhs) : (tensor<2x2x3x2x2xf32>, tensor<2x2x3xf32>) -> tensor<2x2x3x2x3xf32>
    %exp2 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [3] x [1] : (tensor<2x2x3x2x2xf32>, tensor<2x2x3xf32>) -> tensor<2x2x3x2x3xf32>
    check.expect_almost_eq %dr2, %exp2 : tensor<2x2x3x2x3xf32>
    return
  }
}
