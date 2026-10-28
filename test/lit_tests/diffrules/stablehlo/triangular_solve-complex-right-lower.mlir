// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_complex_right_lower outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=triangular_solve_complex_right_lower outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --drop-unsupported-attributes --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --verify-each=0 | stablehlo-translate - --interpret

func.func @triangular_solve_complex_right_lower(%A : tensor<2x2xcomplex<f32>>, %b : tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %y : tensor<2x2xcomplex<f32>>
}

// FORWARD: func.func @triangular_solve_complex_right_lower(%arg0: tensor<2x2xcomplex<f32>>, %arg1: tensor<2x2xcomplex<f32>>, %arg2: tensor<2x2xcomplex<f32>>, %arg3: tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) {
// FORWARD:   %c = stablehlo.constant dense<{{\[\[}}true, false], [true, true]]> : tensor<2x2xi1>
// FORWARD:   %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2x2xcomplex<f32>>
// FORWARD:   %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
// FORWARD:   %1 = stablehlo.select %c, %arg1, %cst : tensor<2x2xi1>, tensor<2x2xcomplex<f32>>
// FORWARD:   %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
// FORWARD:   %3 = stablehlo.subtract %arg3, %2 : tensor<2x2xcomplex<f32>>
// FORWARD:   %4 = "stablehlo.triangular_solve"(%arg0, %3) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
// FORWARD:   return %0, %4 : tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>
// FORWARD: }

// REVERSE: func.func @triangular_solve_complex_right_lower(%arg0: tensor<2x2xcomplex<f32>>, %arg1: tensor<2x2xcomplex<f32>>, %arg2: tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) {
// REVERSE:   %c = stablehlo.constant dense<{{\[\[}}true, false], [true, true]]> : tensor<2x2xi1>
// REVERSE:   %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2x2xcomplex<f32>>
// REVERSE:   %0 = chlo.conj %arg2 : tensor<2x2xcomplex<f32>> -> tensor<2x2xcomplex<f32>>
// REVERSE:   %1 = "stablehlo.triangular_solve"(%arg0, %0) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
// REVERSE:   %2 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
// REVERSE:   %3 = stablehlo.dot_general %2, %1, contracting_dims = [0] x [0] : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
// REVERSE:   %4 = stablehlo.negate %3 : tensor<2x2xcomplex<f32>>
// REVERSE:   %5 = stablehlo.select %c, %4, %cst : tensor<2x2xi1>, tensor<2x2xcomplex<f32>>
// REVERSE:   %6 = chlo.conj %5 : tensor<2x2xcomplex<f32>> -> tensor<2x2xcomplex<f32>>
// REVERSE:   %7 = chlo.conj %1 : tensor<2x2xcomplex<f32>> -> tensor<2x2xcomplex<f32>>
// REVERSE:   return %6, %7 : tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>
// REVERSE: }

func.func @main() {
  %A = stablehlo.constant dense<[[(1.0, 1.0), (0.0, 0.0)], [(2.0, -1.0), (3.0, 2.0)]]> : tensor<2x2xcomplex<f32>>
  %b = stablehlo.constant dense<[[(4.0, 1.0), (5.0, -2.0)], [(6.0, 0.0), (7.0, 3.0)]]> : tensor<2x2xcomplex<f32>>
  %output = stablehlo.constant dense<[[(3.923077, 0.384615), (0.846154, -1.230769)], [(2.538462, 0.307692), (2.076923, -0.384615)]]> : tensor<2x2xcomplex<f32>>

  %dzero = stablehlo.constant dense<(0.0, 0.0)> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 11
  %seed_11 = stablehlo.constant dense<[[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a11:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %seed_11, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a11#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a11#1, dense<[[(-2.153846, 1.769231), (0.0, 0.0)], [(-1.423077, 1.115385), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b11:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %dzero, %b, %seed_11) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b11#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b11#1, dense<[[(0.5, -0.5), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 11
  %rev_11:3 = enzyme.autodiff @triangular_solve_complex_right_lower(%A, %b, %seed_11) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_11#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_11#1, dense<[[(-2.153846, -1.769231), (0.0, 0.0)], [(0.192308, -1.038462), (-0.618343, 0.215976)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_11#2, dense<[[(0.5, 0.5), (0.115385, -0.423077)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 12
  %seed_12 = stablehlo.constant dense<[[(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a12:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %seed_12, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a12#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a12#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b12:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %dzero, %b, %seed_12) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b12#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b12#1, dense<[[(0.115385, 0.423077), (0.230769, -0.153846)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 12
  %rev_12:3 = enzyme.autodiff @triangular_solve_complex_right_lower(%A, %b, %seed_12) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_12#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_12#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (-0.005917, -0.414201)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_12#2, dense<[[(0.0, 0.0), (0.230769, 0.153846)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 21
  %seed_21 = stablehlo.constant dense<[[(0.0, 0.0), (0.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a21:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %seed_21, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a21#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a21#1, dense<[[(0.192308, 1.038462), (0.0, 0.0)], [(-0.846154, 1.230769), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b21:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %dzero, %b, %seed_21) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b21#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b21#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.5, -0.5), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 21
  %rev_21:3 = enzyme.autodiff @triangular_solve_complex_right_lower(%A, %b, %seed_21) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_21#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_21#1, dense<[[(-1.423077, -1.115385), (0.0, 0.0)], [(-0.846154, -1.230769), (-0.402367, 0.83432)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_21#2, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.5, 0.5), (0.115385, -0.423077)]]> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 22
  %seed_22 = stablehlo.constant dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a22:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %seed_22, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a22#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a22#1, dense<[[(-0.618343, -0.215976), (-0.005917, 0.414201)], [(-0.402367, -0.83432), (-0.420118, 0.408284)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b22:2 = enzyme.fwddiff @triangular_solve_complex_right_lower(%A, %dzero, %b, %seed_22) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b22#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b22#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.115385, 0.423077), (0.230769, -0.153846)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 22
  %rev_22:3 = enzyme.autodiff @triangular_solve_complex_right_lower(%A, %b, %seed_22) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_22#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_22#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (-0.420118, -0.408284)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_22#2, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.230769, 0.153846)]]> : tensor<2x2xcomplex<f32>>

  return
}
