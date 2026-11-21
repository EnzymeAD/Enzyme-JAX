// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=pad_dot_general<1>(0);pad_dot_general<1>(1)},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=pad_dot_general<1>(0);pad_dot_general<1>(1)},transform-interpreter,enzyme-hlo-remove-transform)" | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // Reproducer from #1445: pad along both contracting and result dimensions with postPad=1.
  // CHECK-LABEL: func.func private @pad_dot_contracting_and_result
  // CHECK: stablehlo.slice %{{.*}} [0:32, 3:4]
  // CHECK: stablehlo.dot_general %{{.*}}, %{{.*}}, contracting_dims = [0] x [1]{{.*}} : (tensor<1x1xf32>, tensor<32x1xf32>) -> tensor<1x32xf32>
  // CHECK: stablehlo.pad %{{.*}}, %{{.*}}, low = [2, 0], high = [29, 0], interior = [0, 0] : (tensor<1x32xf32>, tensor<f32>) -> tensor<32x32xf32>
  func.func private @pad_dot_contracting_and_result(%arg0: tensor<f32>) -> (tensor<32x32xf32>, tensor<f32>) {
    %c = stablehlo.constant dense<3> : tensor<32x32xi64>
    %c_0 = stablehlo.constant dense<2> : tensor<32x32xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.pad %0, %cst, low = [2, 3], high = [29, 28], interior = [0, 0] : (tensor<1x1xf32>, tensor<f32>) -> tensor<32x32xf32>
    %2 = stablehlo.convert %c_0 : (tensor<32x32xi64>) -> tensor<32x32xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<32x32xf32>
    %4 = stablehlo.convert %c : (tensor<32x32xi64>) -> tensor<32x32xf32>
    %5 = stablehlo.subtract %3, %4 : tensor<32x32xf32>
    %6 = stablehlo.pad %0, %cst, low = [3, 2], high = [28, 29], interior = [0, 0] : (tensor<1x1xf32>, tensor<f32>) -> tensor<32x32xf32>
    %7 = stablehlo.dot_general %6, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %7, %arg0 : tensor<32x32xf32>, tensor<f32>
  }

  // Also test padded batching + contracting + result dimensions with postPad=1.
  // CHECK-LABEL: func.func private @pad_dot_batch_and_contracting
  // CHECK: stablehlo.slice %{{.*}} [1:3, 1:3, 0:2]
  // CHECK: stablehlo.dot_general {{.*}} batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x1x2xf32>, tensor<2x2x2xf32>) -> tensor<2x1x2xf32>
  // CHECK: stablehlo.pad {{.*}} low = [1, 1, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<2x1x2xf32>, tensor<f32>) -> tensor<4x2x2xf32>
  func.func private @pad_dot_batch_and_contracting(%x: tensor<2x1x2xf32>, %y: tensor<4x4x2xf32>) -> tensor<4x2x2xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %px = stablehlo.pad %x, %z, low = [1, 1, 1], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<2x1x2xf32>, tensor<f32>) -> tensor<4x2x4xf32>
    %d = stablehlo.dot_general %px, %y, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x2x4xf32>, tensor<4x4x2xf32>) -> tensor<4x2x2xf32>
    return %d : tensor<4x2x2xf32>
  }

  func.func @main() {
    %a = stablehlo.constant dense<5.0> : tensor<f32>
    %r:2 = func.call @pad_dot_contracting_and_result(%a) : (tensor<f32>) -> (tensor<32x32xf32>, tensor<f32>)
    // At row 2, col 2: %1[2,3] = 5 -> %5[2,3] = 2*5 - 3 = 7, all other %5[r,3] = -3;
    // %6[3,2] = 5 -> %7[2,2] = 5 * 7 = 35, %7[2, r!=2] = 5 * (-3) = -15, other rows = 0.
    %row2_col2 = stablehlo.slice %r#0 [2:3, 2:3] : (tensor<32x32xf32>) -> tensor<1x1xf32>
    check.expect_almost_eq_const %row2_col2, dense<[[35.0]]> : tensor<1x1xf32>
    %row2_col0 = stablehlo.slice %r#0 [2:3, 0:1] : (tensor<32x32xf32>) -> tensor<1x1xf32>
    check.expect_almost_eq_const %row2_col0, dense<[[-15.0]]> : tensor<1x1xf32>
    %row0_col2 = stablehlo.slice %r#0 [0:1, 2:3] : (tensor<32x32xf32>) -> tensor<1x1xf32>
    check.expect_almost_eq_const %row0_col2, dense<[[0.0]]> : tensor<1x1xf32>

    %x = stablehlo.constant dense<[[[1.0, 2.0]], [[3.0, -1.0]]]> : tensor<2x1x2xf32>
    %y = stablehlo.constant dense<1.0> : tensor<4x4x2xf32>
    %b = func.call @pad_dot_batch_and_contracting(%x, %y) : (tensor<2x1x2xf32>, tensor<4x4x2xf32>) -> tensor<4x2x2xf32>
    check.expect_almost_eq_const %b, dense<[
      [[0.0, 0.0], [0.0, 0.0]],
      [[0.0, 0.0], [3.0, 3.0]],
      [[0.0, 0.0], [2.0, 2.0]],
      [[0.0, 0.0], [0.0, 0.0]]
    ]> : tensor<4x2x2xf32>
    return
  }
}
