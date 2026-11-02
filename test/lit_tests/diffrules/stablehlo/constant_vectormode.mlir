// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | stablehlo-translate - --interpret --allow-unregistered-dialect

// At vector width > 1 a derivative rule mixes batched values (the tangent or
// adjoint) with primal-shaped ones (operands, literal constants such as the 1
// in d tanh = 1 - tanh^2). The primal-shaped part is computed once at primal
// shape and broadcast only where it meets a batched value; previously the
// literal constant kept its primal shape while everything else was batched,
// and building subtract(tensor<3xf32>, tensor<2x3xf32>) aborted compilation.

func.func @tanh(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.tanh %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

func.func @logistic(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.logistic %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

func.func @log_plus_one(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.log_plus_one %x : (tensor<3xf32>) -> tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// CHECK-LABEL: func.func private @fwddiffe2tanh(
// CHECK-SAME:      %[[X:.+]]: tensor<3xf32>, %[[DX:.+]]: tensor<2x3xf32>)
// CHECK:         %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:         %[[T:.+]] = stablehlo.tanh %[[X]] : tensor<3xf32>
// CHECK:         %[[T2:.+]] = stablehlo.multiply %[[T]], %[[T]] : tensor<3xf32>
// CHECK:         %[[D:.+]] = stablehlo.subtract %[[ONE]], %[[T2]] : tensor<3xf32>
// CHECK:         %[[B:.+]] = stablehlo.broadcast_in_dim %[[D]], dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[R:.+]] = stablehlo.multiply %[[DX]], %[[B]] : tensor<2x3xf32>
// CHECK:         return %[[R]]

// CHECK-LABEL: func.func private @fwddiffe2logistic(
// CHECK-SAME:      %[[X:.+]]: tensor<3xf32>, %[[DX:.+]]: tensor<2x3xf32>)
// CHECK:         %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:         %[[S:.+]] = stablehlo.logistic %[[X]] : tensor<3xf32>
// CHECK:         %[[OMS:.+]] = stablehlo.subtract %[[ONE]], %[[S]] : tensor<3xf32>
// CHECK:         %[[D:.+]] = stablehlo.multiply %[[S]], %[[OMS]] : tensor<3xf32>
// CHECK:         %[[B:.+]] = stablehlo.broadcast_in_dim %[[D]], dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[R:.+]] = stablehlo.multiply %[[DX]], %[[B]] : tensor<2x3xf32>
// CHECK:         return %[[R]]

// CHECK-LABEL: func.func private @fwddiffe2log_plus_one(
// CHECK-SAME:      %[[X:.+]]: tensor<3xf32>, %[[DX:.+]]: tensor<2x3xf32>)
// CHECK:         %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:         %[[XP1:.+]] = stablehlo.add %[[X]], %[[ONE]] : tensor<3xf32>
// CHECK:         %[[B:.+]] = stablehlo.broadcast_in_dim %[[XP1]], dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[R:.+]] = stablehlo.divide %[[DX]], %[[B]] : tensor<2x3xf32>
// CHECK:         return %[[R]]

// With a strong zero the CheckedMul subroutine compares the batched tangent
// against a literal zero, which is brought up to the batched shape at the
// compare while the primal-only part of the rule stays primal-shaped.
// CHECK-LABEL: func.func private @fwddiffe2tanh_0(
// CHECK-SAME:      %[[X:.+]]: tensor<3xf32>, %[[DX:.+]]: tensor<2x3xf32>)
// CHECK-DAG:     %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK-DAG:     %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:         %[[T:.+]] = stablehlo.tanh %[[X]] : tensor<3xf32>
// CHECK:         %[[T2:.+]] = stablehlo.multiply %[[T]], %[[T]] : tensor<3xf32>
// CHECK:         %[[D:.+]] = stablehlo.subtract %[[ONE]], %[[T2]] : tensor<3xf32>
// CHECK:         %[[ISZ:.+]] = stablehlo.compare EQ, %[[DX]], %[[ZERO]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
// CHECK:         %[[B:.+]] = stablehlo.broadcast_in_dim %[[D]], dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[M:.+]] = stablehlo.multiply %[[DX]], %[[B]] : tensor<2x3xf32>
// CHECK:         %[[R:.+]] = stablehlo.select %[[ISZ]], %[[ZERO]], %[[M]] : tensor<2x3xi1>, tensor<2x3xf32>
// CHECK:         return %[[R]]

// CHECK-LABEL: func.func private @diffe2tanh(
// CHECK-SAME:      %[[X:.+]]: tensor<3xf32>, %[[DR:.+]]: tensor<2x3xf32>)
// CHECK:         %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK:         %[[T:.+]] = stablehlo.tanh %[[X]] : tensor<3xf32>
// CHECK:         %[[T2:.+]] = stablehlo.multiply %[[T]], %[[T]] : tensor<3xf32>
// CHECK:         %[[D:.+]] = stablehlo.subtract %[[ONE]], %[[T2]] : tensor<3xf32>
// CHECK:         %[[B:.+]] = stablehlo.broadcast_in_dim %[[D]], dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[R:.+]] = stablehlo.multiply %[[DR]], %[[B]] : tensor<2x3xf32>
// CHECK:         return %[[R]]

func.func @main() {
  %x = stablehlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf32>
  // Two tangents: ones, and (1, 2, 0).
  %dx = stablehlo.constant dense<[[1.0, 1.0, 1.0], [1.0, 2.0, 0.0]]> : tensor<2x3xf32>

  // tanh'(x) = 1 - tanh(x)^2
  %tanh_expected = stablehlo.constant dense<[[1.0, 0.41997434161404607, 0.4199743416139812], [1.0, 0.8399486832280921, 0.0]]> : tensor<2x3xf32>
  %tanh_fwd = enzyme.fwddiff @tanh(%x, %dx) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    width = 2 : i64
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq %tanh_fwd, %tanh_expected : tensor<2x3xf32>

  // logistic'(x) = logistic(x) * (1 - logistic(x))
  %logistic_expected = stablehlo.constant dense<[[0.25, 0.19661193324148185, 0.19661193324148185], [0.25, 0.3932238664829637, 0.0]]> : tensor<2x3xf32>
  %logistic_fwd = enzyme.fwddiff @logistic(%x, %dx) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    width = 2 : i64
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq %logistic_fwd, %logistic_expected : tensor<2x3xf32>

  // log1p'(x) = 1 / (1 + x); use x = (0, 1, 3) to stay in the domain.
  %x1 = stablehlo.constant dense<[0.0, 1.0, 3.0]> : tensor<3xf32>
  %log1p_expected = stablehlo.constant dense<[[1.0, 0.5, 0.25], [1.0, 1.0, 0.0]]> : tensor<2x3xf32>
  %log1p_fwd = enzyme.fwddiff @log_plus_one(%x1, %dx) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    width = 2 : i64
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq %log1p_fwd, %log1p_expected : tensor<2x3xf32>

  %tanh_fwd_sz = enzyme.fwddiff @tanh(%x, %dx) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    width = 2 : i64,
    strong_zero = true
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq %tanh_fwd_sz, %tanh_expected : tensor<2x3xf32>

  // Reverse mode at width 2 goes through the same rule.
  %tanh_rev = enzyme.autodiff @tanh(%x, %dx) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_activenoneed>],
    width = 2 : i64
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq %tanh_rev, %tanh_expected : tensor<2x3xf32>

  func.return
}
