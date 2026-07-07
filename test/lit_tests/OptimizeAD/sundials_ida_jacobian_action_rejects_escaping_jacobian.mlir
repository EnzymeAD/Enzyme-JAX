// RUN: not enzymexlamlir-opt --lower-sundials-ida-jacobian-action-stablehlo %s 2>&1 | FileCheck %s

module {
  func.func @residual(%x: tensor<2xf64>) -> tensor<2xf64> {
    return %x : tensor<2xf64>
  }

  func.func @returns_full_jacobian(%x: tensor<2xf64>) -> tensor<2x2xf64> {
    %j = enzyme.jacobian @residual(%x) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dup>]
    } : (tensor<2xf64>) -> tensor<2x2xf64>
    return %j : tensor<2x2xf64>
  }

  func.func @solve(%x: tensor<2xf64>) -> tensor<2x2xf64> {
    %state = enzymexla.sundials.ida_solve residual = @residual
      jacobian_action = @returns_full_jacobian
      linear_solver = <jacobian_action_iterative>
      jacobian_demand = <jacobian_action>
      (%x) : (tensor<2xf64>) -> tensor<2x2xf64>
    return %state : tensor<2x2xf64>
  }
}

// CHECK: unsupported live Jacobian materialization remains after SUNDIALS IDA Jacobian-action lowering
