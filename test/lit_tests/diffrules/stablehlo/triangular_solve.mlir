// RUN enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// TODO-RUN enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @triangular_solve(%A : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %y : tensor<2x2xf32>
}

// FORWARD: func.func @triangular_solve(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// FORWARD-NEXT:     %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-NEXT:     %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-NEXT:     %2 = stablehlo.subtract %arg3, %1 : tensor<2x2xf32>
// FORWARD-NEXT:     %3 = "stablehlo.triangular_solve"(%arg0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-NEXT:     return %0, %3 : tensor<2x2xf32>, tensor<2x2xf32>
// FORWARD-NEXT: }

func.func @main() {
  %A = stablehlo.constant dense<[[1.0, 0.0], [2.0, 3.0]]> : tensor<2x2xf32>
  %b = stablehlo.constant dense<[[4.0, 5.0], [6.0, 7.0]]> : tensor<2x2xf32>
  %output = stablehlo.constant dense<[[4.0, 5.0], [-0.666667, -1.0]]> : tensor<2x2xf32>

  %dzero = stablehlo.constant dense<0.0> : tensor<2x2xf32>

  %dA_11 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_11:2 = enzyme.fwddiff @triangular_solve(%A, %dA_11, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_11#1, dense<[[-4.0, -5.0], [2.66667, 3.33333]]> : tensor<2x2xf32>

  %dA_12 = stablehlo.constant dense<[[0.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_12:2 = enzyme.fwddiff @triangular_solve(%A, %dA_12, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_12#1, dense<0.0> : tensor<2x2xf32>

  %dA_21 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_21:2 = enzyme.fwddiff @triangular_solve(%A, %dA_21, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_21#1, dense<[[0.0, 0.0], [-1.33333, -1.66667]]> : tensor<2x2xf32>

  %dA_22 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_22:2 = enzyme.fwddiff @triangular_solve(%A, %dA_22, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_22#1, dense<[[0.0, 0.0], [0.222222, 0.333333]]> : tensor<2x2xf32>

  return
}