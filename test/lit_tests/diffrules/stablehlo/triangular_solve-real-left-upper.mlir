// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_left_upper outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_left_upper outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --verify-each=0 | stablehlo-translate - --interpret

func.func @triangular_solve_real_left_upper(%A : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %y : tensor<2x2xf32>
}

// FORWARD: func.func @triangular_solve_real_left_upper(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// FORWARD:   %c = stablehlo.constant dense<{{\[\[}}true, true], [false, true]]> : tensor<2x2xi1>
// FORWARD:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// FORWARD:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD:   %1 = stablehlo.select %c, %arg1, %cst : tensor<2x2xi1>, tensor<2x2xf32>
// FORWARD:   %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD:   %3 = stablehlo.subtract %arg3, %2 : tensor<2x2xf32>
// FORWARD:   %4 = "stablehlo.triangular_solve"(%arg0, %3) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD:   return %0, %4 : tensor<2x2xf32>, tensor<2x2xf32>
// FORWARD: }

// REVERSE: func.func @triangular_solve_real_left_upper(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// REVERSE:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// REVERSE:   %c = stablehlo.constant dense<{{\[\[}}true, true], [false, true]]> : tensor<2x2xi1>
// REVERSE:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// REVERSE:   %1 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// REVERSE:   %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [1] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// REVERSE:   %3 = stablehlo.negate %2 : tensor<2x2xf32>
// REVERSE:   %4 = stablehlo.select %c, %3, %cst : tensor<2x2xi1>, tensor<2x2xf32>
// REVERSE:   return %4, %0 : tensor<2x2xf32>, tensor<2x2xf32>
// REVERSE: }

func.func @main() {
  %A = stablehlo.constant dense<[[1.0, 2.0], [0.0, 3.0]]> : tensor<2x2xf32>
  %b = stablehlo.constant dense<[[4.0, 5.0], [6.0, 7.0]]> : tensor<2x2xf32>
  %output = stablehlo.constant dense<[[0.0, 0.333333], [2.0, 2.333333]]> : tensor<2x2xf32>

  %dzero = stablehlo.constant dense<0.0> : tensor<2x2xf32>

  // fwd, ij = 11
  %seed_11 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a11:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %seed_11, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a11#1, dense<[[0.0, -0.333333], [0.0, 0.0]]> : tensor<2x2xf32>

  %fwd_b11:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %dzero, %b, %seed_11) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b11#1, dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>

  // rev, ij = 11
  %rev_11:3 = enzyme.autodiff @triangular_solve_real_left_upper(%A, %b, %seed_11) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_11#1, dense<[[0.0, -2.0], [0.0, 1.333333]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_11#2, dense<[[1.0, 0.0], [-0.666667, 0.0]]> : tensor<2x2xf32>

  // fwd, ij = 12
  %seed_12 = stablehlo.constant dense<[[0.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a12:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %seed_12, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a12#1, dense<[[-2.0, -2.333333], [0.0, 0.0]]> : tensor<2x2xf32>

  %fwd_b12:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %dzero, %b, %seed_12) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b12#1, dense<[[0.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>

  // rev, ij = 12
  %rev_12:3 = enzyme.autodiff @triangular_solve_real_left_upper(%A, %b, %seed_12) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_12#1, dense<[[-0.333333, -2.333333], [0.0, 1.555556]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_12#2, dense<[[0.0, 1.0], [0.0, -0.666667]]> : tensor<2x2xf32>

  // fwd, ij = 21
  %seed_21 = stablehlo.constant dense<[[0.0, 0.0], [1.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a21:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %seed_21, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a21#1, dense<[[0.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>

  %fwd_b21:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %dzero, %b, %seed_21) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b21#1, dense<[[-0.666667, 0.0], [0.333333, 0.0]]> : tensor<2x2xf32>

  // rev, ij = 21
  %rev_21:3 = enzyme.autodiff @triangular_solve_real_left_upper(%A, %b, %seed_21) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_21#1, dense<[[0.0, 0.0], [0.0, -0.666667]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_21#2, dense<[[0.0, 0.0], [0.333333, 0.0]]> : tensor<2x2xf32>

  // fwd, ij = 22
  %seed_22 = stablehlo.constant dense<[[0.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
  %fwd_a22:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %seed_22, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a22#1, dense<[[1.333333, 1.555556], [-0.666667, -0.777778]]> : tensor<2x2xf32>

  %fwd_b22:2 = enzyme.fwddiff @triangular_solve_real_left_upper(%A, %dzero, %b, %seed_22) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b22#1, dense<[[0.0, -0.666667], [0.0, 0.333333]]> : tensor<2x2xf32>

  // rev, ij = 22
  %rev_22:3 = enzyme.autodiff @triangular_solve_real_left_upper(%A, %b, %seed_22) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_22#1, dense<[[0.0, 0.0], [0.0, -0.777778]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_22#2, dense<[[0.0, 0.0], [0.0, 0.333333]]> : tensor<2x2xf32>

  return
}
