// RUN: enzymexlamlir-opt --split-input-file --lower-gridkit-ida-jacobian-action-stablehlo %s | FileCheck %s

module {
  func.func @residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @ida_jac_times_path(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %j = enzyme.jacobian @residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %r = stablehlo.dot_general %j, %v, contracting_dims = [1] x [0] {gridkit.solver = "ida_jac_times"} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @ida_jac_times_path
// CHECK-SAME: (%[[X:.*]]: tensor<2xf64>, %[[V:.*]]: tensor<2xf64>) -> tensor<2xf64>
// CHECK:         %[[J:.*]] = enzyme.jacobian @residual(%[[X]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dup>]} : (tensor<2xf64>) -> tensor<2x2xf64>
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[RES:.*]] = enzyme.fwddiff @residual(%[[X]], %[[V]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<2xf64>

// -----

module {
  func.func @residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @direct_sparse_path(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %j = enzyme.jacobian @residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %r = stablehlo.dot_general %j, %v, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @direct_sparse_path
// CHECK:         %[[J:.*]] = enzyme.jacobian
// CHECK:         stablehlo.dot_general %[[J]]
// CHECK-NOT:     enzyme.fwddiff @residual
