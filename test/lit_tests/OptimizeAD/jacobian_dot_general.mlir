// RUN: enzymexlamlir-opt --split-input-file --lower-enzyme-jacobian-stablehlo %s | FileCheck %s

module {
  func.func @id(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @test_jvp(%x: tensor<2xf64>, %dx: tensor<2xf64>) -> tensor<2xf64> {
    %j = enzyme.jacobian @id(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %r = stablehlo.dot_general %j, %dx, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @test_jvp
// CHECK-SAME: (%[[X:.*]]: tensor<2xf64>, %[[DX:.*]]: tensor<2xf64>) -> tensor<2xf64>
// CHECK:         %[[J:.*]] = enzyme.jacobian @id(%[[X]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dup>]} : (tensor<2xf64>) -> tensor<2x2xf64>
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[RES:.*]] = enzyme.fwddiff @id(%[[X]], %[[DX]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<2xf64>

// -----

module {
  func.func @expand(%x: tensor<1xf64>) -> tensor<1x2xf64> {
    %0 = stablehlo.broadcast_in_dim %x, dims = [0] : (tensor<1xf64>) -> tensor<1x2xf64>
    return %0 : tensor<1x2xf64>
  }

  func.func @test_vjp(%x: tensor<1xf64>, %dy: tensor<2x1xf64>) -> tensor<1xf64> {
    %j = enzyme.jacobian @expand(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<1xf64>) -> tensor<1x2x1xf64>
    %r = stablehlo.dot_general %dy, %j, contracting_dims = [0, 1] x [1, 2] : (tensor<2x1xf64>, tensor<1x2x1xf64>) -> tensor<1xf64>
    return %r : tensor<1xf64>
  }
}

// CHECK-LABEL: func.func @test_vjp
// CHECK-SAME: (%[[X:.*]]: tensor<1xf64>, %[[DY:.*]]: tensor<2x1xf64>) -> tensor<1xf64>
// CHECK:         %[[J:.*]] = enzyme.jacobian @expand(%[[X]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dup>]} : (tensor<1xf64>) -> tensor<1x2x1xf64>
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[RES:.*]] = enzyme.autodiff @expand(%[[X]], %[[DY]]) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<1xf64>, tensor<2x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<1xf64>
