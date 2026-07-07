// RUN: not enzymexlamlir-opt --lower-sundials-ida-jacobian-action-stablehlo %s 2>&1 | FileCheck %s

module {
  func.func @residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @inspects_matrix_element(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<1x1xf64> {
    %j = enzyme.jacobian @residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    %elt = stablehlo.slice %j [0:1, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
    return %elt : tensor<1x1xf64>
  }

  func.func @solve(%x: tensor<2xf64>, %v: tensor<2xf64>) -> tensor<2xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian_action = @inspects_matrix_element
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%x, %v) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %state : tensor<2xf64>
  }
}

// CHECK: unsupported live Jacobian materialization remains after SUNDIALS IDA Jacobian-action lowering
