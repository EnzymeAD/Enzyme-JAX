// RUN: enzymexlamlir-opt --split-input-file --lower-sundials-ida-jacobian-action-stablehlo %s | FileCheck %s

module {
  func.func @residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @jactimes(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %j = enzyme.jacobian @residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %r = stablehlo.dot_general %j, %v, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }

  func.func @solve(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian_action = @jactimes
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%x, %v) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %state : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @jactimes
// CHECK-SAME: (%[[X:.*]]: tensor<2xf64>, %[[V:.*]]: tensor<2xf64>) -> tensor<2xf64>
// CHECK-NOT:     enzyme.jacobian
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[RES:.*]] = enzyme.fwddiff @residual(%[[X]], %[[V]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<2xf64>

// -----

module {
  func.func @residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @jactimes(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %j = enzyme.jacobian @residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %r = stablehlo.dot_general %j, %v, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }

  func.func @solve(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian_action = @jactimes
      linear_solver = <explicit_sparse_direct>
      jacobian_demand = <explicit_matrix>
      (%x, %v) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %state : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @jactimes
// CHECK:         %[[J:.*]] = enzyme.jacobian
// CHECK:         stablehlo.dot_general %[[J]]
// CHECK-NOT:     enzyme.fwddiff @residual

// -----

module {
  func.func @ida_residual(%y: tensor<2xf64>, %yp: tensor<2xf64>) -> tensor<2xf64> {
    return %y : tensor<2xf64>
  }

  func.func @ida_jactimes(%y: tensor<2xf64>, %yp: tensor<2xf64>, %v: tensor<2xf64>, %cjv: tensor<2xf64>) -> tensor<2xf64> {
    %jy, %jyp = enzyme.jacobian @ida_residual(%y, %yp) {
      activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2x2xf64>, tensor<2x2xf64>)
    %dy = stablehlo.dot_general %jy, %v, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %dyp = stablehlo.dot_general %jyp, %cjv, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %r = stablehlo.add %dy, %dyp : tensor<2xf64>
    return %r : tensor<2xf64>
  }

  func.func @solve(%y: tensor<2xf64>, %yp: tensor<2xf64>, %v: tensor<2xf64>, %cjv: tensor<2xf64>) -> tensor<2xf64> {
    %state = enzymexla.sundials.ida_solve residual = @ida_residual
      jacobian_action = @ida_jactimes
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%y, %yp, %v, %cjv) : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %state : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @ida_jactimes
// CHECK-SAME: (%[[Y:.*]]: tensor<2xf64>, %[[YP:.*]]: tensor<2xf64>, %[[V:.*]]: tensor<2xf64>, %[[CJV:.*]]: tensor<2xf64>) -> tensor<2xf64>
// CHECK-NOT:     enzyme.jacobian
// CHECK-NOT:     stablehlo.dot_general
// CHECK-NOT:     stablehlo.add
// CHECK:         %[[RES:.*]] = enzyme.fwddiff @ida_residual(%[[Y]], %[[V]], %[[YP]], %[[CJV]]) {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<2xf64>

// -----

module {
  func.func @dynamic_residual(%x: tensor<?xf64>) -> tensor<?xf64> {
    return %x : tensor<?xf64>
  }

  func.func @dynamic_jactimes(%x: tensor<?xf64>, %v: tensor<?xf64>) -> tensor<?xf64> {
    %j = enzyme.jacobian @dynamic_residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<?xf64>) -> tensor<?x?xf64>
    %r = stablehlo.dot_general %j, %v, contracting_dims = [1] x [0] : (tensor<?x?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %r : tensor<?xf64>
  }

  func.func @solve(%x: tensor<?xf64>, %v: tensor<?xf64>) -> tensor<?xf64> {
    %state = enzymexla.sundials.ida_solve residual = @dynamic_residual
      jacobian_action = @dynamic_jactimes
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%x, %v) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %state : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @dynamic_jactimes
// CHECK-SAME: (%[[X:.*]]: tensor<?xf64>, %[[V:.*]]: tensor<?xf64>) -> tensor<?xf64>
// CHECK-NOT:     enzyme.jacobian
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[RES:.*]] = enzyme.fwddiff @dynamic_residual(%[[X]], %[[V]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
// CHECK-NEXT:    return %[[RES]] : tensor<?xf64>

// -----

module {
  func.func @multi_action_residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @two_jactimes(%x: tensor<2xf64>, %v0: tensor<2xf64>, %v1: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
    %j = enzyme.jacobian @multi_action_residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %r0 = stablehlo.dot_general %j, %v0, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %r1 = stablehlo.dot_general %j, %v1, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %r0, %r1 : tensor<2xf64>, tensor<2xf64>
  }

  func.func @solve(%x: tensor<2xf64>, %v0: tensor<2xf64>, %v1: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
    %state:2 = enzymexla.sundials.ida_solve residual = @multi_action_residual
      jacobian_action = @two_jactimes
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%x, %v0, %v1) : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)
    return %state#0, %state#1 : tensor<2xf64>, tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @two_jactimes
// CHECK-SAME: (%[[X:.*]]: tensor<2xf64>, %[[V0:.*]]: tensor<2xf64>, %[[V1:.*]]: tensor<2xf64>)
// CHECK-NOT:     enzyme.jacobian
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[R0:.*]] = enzyme.fwddiff @multi_action_residual(%[[X]], %[[V0]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK:         %[[R1:.*]] = enzyme.fwddiff @multi_action_residual(%[[X]], %[[V1]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    return %[[R0]], %[[R1]] : tensor<2xf64>, tensor<2xf64>
