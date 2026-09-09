// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_left_lower outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-REAL-LEFT-LOWER
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_left_upper outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-REAL-LEFT-UPPER
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_right_lower outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-REAL-RIGHT-LOWER
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_right_upper outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-REAL-RIGHT-UPPER

// TODO-RUN enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_real_left_lower outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// TODO-RUN enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @triangular_solve_real_left_lower(%A : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %y : tensor<2x2xf32>
}

// FORWARD-REAL-LEFT-LOWER: func.func @triangular_solve_real_left_lower(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// FORWARD-REAL-LEFT-LOWER:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// FORWARD-REAL-LEFT-LOWER:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-LEFT-LOWER:   %1 = enzymexla.blas.trmm %arg1, %0, %cst {side = #enzymexla.side<left>, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<L>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// FORWARD-REAL-LEFT-LOWER:   %2 = stablehlo.subtract %arg3, %1 : tensor<2x2xf32>
// FORWARD-REAL-LEFT-LOWER:   %3 = "stablehlo.triangular_solve"(%arg0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-LEFT-LOWER:   return %0, %3 : tensor<2x2xf32>, tensor<2x2xf32>
// FORWARD-REAL-LEFT-LOWER: }

func.func @triangular_solve_real_left_upper(%A : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %y : tensor<2x2xf32>
}

// FORWARD-REAL-LEFT-UPPER: func.func @triangular_solve_real_left_upper(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// FORWARD-REAL-LEFT-UPPER:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// FORWARD-REAL-LEFT-UPPER:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-LEFT-UPPER:   %1 = enzymexla.blas.trmm %arg1, %0, %cst {side = #enzymexla.side<left>, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// FORWARD-REAL-LEFT-UPPER:   %2 = stablehlo.subtract %arg3, %1 : tensor<2x2xf32>
// FORWARD-REAL-LEFT-UPPER:   %3 = "stablehlo.triangular_solve"(%arg0, %2) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-LEFT-UPPER:   return %0, %3 : tensor<2x2xf32>, tensor<2x2xf32>
// FORWARD-REAL-LEFT-UPPER: }

func.func @triangular_solve_real_right_lower(%A : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %y : tensor<2x2xf32>
}

// FORWARD-REAL-RIGHT-LOWER: func.func @triangular_solve_real_right_lower(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// FORWARD-REAL-RIGHT-LOWER:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// FORWARD-REAL-RIGHT-LOWER:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-RIGHT-LOWER:   %1 = enzymexla.blas.trmm %arg1, %0, %cst {side = #enzymexla.side<right>, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<L>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// FORWARD-REAL-RIGHT-LOWER:   %2 = stablehlo.subtract %arg3, %1 : tensor<2x2xf32>
// FORWARD-REAL-RIGHT-LOWER:   %3 = "stablehlo.triangular_solve"(%arg0, %2) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-RIGHT-LOWER:   return %0, %3 : tensor<2x2xf32>, tensor<2x2xf32>
// FORWARD-REAL-RIGHT-LOWER: }

func.func @triangular_solve_real_right_upper(%A : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = false, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %y : tensor<2x2xf32>
}

// FORWARD-REAL-RIGHT-UPPER: func.func @triangular_solve_real_right_upper(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// FORWARD-REAL-RIGHT-UPPER:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// FORWARD-REAL-RIGHT-UPPER:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-RIGHT-UPPER:   %1 = enzymexla.blas.trmm %arg1, %0, %cst {side = #enzymexla.side<right>, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// FORWARD-REAL-RIGHT-UPPER:   %2 = stablehlo.subtract %arg3, %1 : tensor<2x2xf32>
// FORWARD-REAL-RIGHT-UPPER:   %3 = "stablehlo.triangular_solve"(%arg0, %2) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FORWARD-REAL-RIGHT-UPPER:   return %0, %3 : tensor<2x2xf32>, tensor<2x2xf32>
// FORWARD-REAL-RIGHT-UPPER: }

func.func @num_test_real_left_lower() {
  %A = stablehlo.constant dense<[[1.0, 0.0], [2.0, 3.0]]> : tensor<2x2xf32>
  %b = stablehlo.constant dense<[[4.0, 5.0], [6.0, 7.0]]> : tensor<2x2xf32>
  %output = stablehlo.constant dense<[[4.0, 5.0], [-0.666667, -1.0]]> : tensor<2x2xf32>

  %dzero = stablehlo.constant dense<0.0> : tensor<2x2xf32>

  // fwd, ij = 11
  %seed_11 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a11:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %seed_11, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a11#1, dense<[[-4.0, -5.0], [2.66667, 3.33333]]> : tensor<2x2xf32>

  %fwd_b11:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %dzero, %b, %seed_11) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b11#1, dense<[[1.0, 0.0], [-0.666667, 0.0]]> : tensor<2x2xf32>

  // rev, ij = 11
  %rev_11:3 = enzyme.autodiff @triangular_solve_real_left_lower(%A, %b, %seed_11) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_11#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_11#1, dense<[[-4.0, 0.666667], [0.0, 0.0]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_11#2, dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>

  // fwd, ij = 12
  %seed_12 = stablehlo.constant dense<[[0.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a12:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %seed_12, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a12#1, dense<[[0.666667, 1.0], [-0.444444, -0.666667]]> : tensor<2x2xf32>

  %fwd_b12:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %dzero, %b, %seed_12) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b12#1, dense<[[0.0, 1.0], [0.0, -0.666667]]> : tensor<2x2xf32>

  // rev, ij = 12
  %rev_12:3 = enzyme.autodiff @triangular_solve_real_left_lower(%A, %b, %seed_12) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_12#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_12#1, dense<[[-5.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_12#2, dense<[[0.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>

  // fwd, ij = 21
  %seed_21 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a21:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %seed_21, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a21#1, dense<[[0.0, 0.0], [-1.33333, -1.66667]]> : tensor<2x2xf32>

  %fwd_b21:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %dzero, %b, %seed_21) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b21#1, dense<[[0.0, 0.0], [0.333333, 0.0]]> : tensor<2x2xf32>

  // rev, ij = 21
  %rev_21:3 = enzyme.autodiff @triangular_solve_real_left_lower(%A, %b, %seed_21) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_21#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_21#1, dense<[[2.66667, -0.444444], [-1.33333, 0.222222]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_21#2, dense<[[-0.666667, 0.0], [0.333333, 0.0]]> : tensor<2x2xf32>

  // fwd, ij = 22
  %seed_22 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  %fwd_a22:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %seed_22, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_a22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_a22#1, dense<[[0.0, 0.0], [0.222222, 0.333333]]> : tensor<2x2xf32>

  %fwd_b22:2 = enzyme.fwddiff @triangular_solve_real_left_lower(%A, %dzero, %b, %seed_22) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %fwd_b22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %fwd_b22#1, dense<[[0.0, 0.0], [0.0, 0.333333]]> : tensor<2x2xf32>

  // rev, ij = 22
  %rev_22:3 = enzyme.autodiff @triangular_solve_real_left_lower(%A, %b, %seed_22) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  check.expect_almost_eq %rev_22#0, %output : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_22#1, dense<[[3.33333, -0.666667], [-1.66667, 0.333333]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev_22#2, dense<[[0.0, -0.666667], [0.0, 0.333333]]> : tensor<2x2xf32>

  return
}