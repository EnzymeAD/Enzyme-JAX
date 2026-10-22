// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --verify-each=0 | stablehlo-translate - --interpret

func.func @triangular_solve_complex_left_lower(%A : tensor<2x2xcomplex<f32>>, %b : tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %y = "stablehlo.triangular_solve"(%A, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %y : tensor<2x2xcomplex<f32>>
}

func.func @main() {
  %A = stablehlo.constant dense<[[(1.0, 1.0), (0.0, 0.0)], [(2.0, -1.0), (3.0, 2.0)]]> : tensor<2x2xcomplex<f32>>
  %b = stablehlo.constant dense<[[(4.0, 1.0), (5.0, -2.0)], [(6.0, 0.0), (7.0, 3.0)]]> : tensor<2x2xcomplex<f32>>
  %output = stablehlo.constant dense<[[(2.5, -1.5), (1.5, -3.5)], [(1.423077, 0.884615), (3.5, 1.5)]]> : tensor<2x2xcomplex<f32>>

  %dzero = stablehlo.constant dense<(0.0, 0.0)> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 11
  %seed_11 = stablehlo.constant dense<[[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a11:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %seed_11, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a11#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a11#1, dense<[[(-0.5, 2.0), (1.0, 2.5)], [(-0.923077, -0.884615), (-1.653846, -0.230769)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b11:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %dzero, %b, %seed_11) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b11#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b11#1, dense<[[(0.5, -0.5), (0.0, 0.0)], [(0.115385, 0.423077), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 11
  %rev_11:3 = enzyme.autodiff @triangular_solve_complex_left_lower(%A, %b, %seed_11) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_11#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_11#1, dense<[[(-0.5, -2.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_11#2, dense<[[(0.5, 0.5), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 12
  %seed_12 = stablehlo.constant dense<[[(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a12:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %seed_12, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a12#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a12#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b12:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %dzero, %b, %seed_12) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b12#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b12#1, dense<[[(0.0, 0.0), (0.5, -0.5)], [(0.0, 0.0), (0.115385, 0.423077)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 12
  %rev_12:3 = enzyme.autodiff @triangular_solve_complex_left_lower(%A, %b, %seed_12) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_12#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_12#1, dense<[[(1.0, -2.5), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_12#2, dense<[[(0.0, 0.0), (0.5, 0.5)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 21
  %seed_21 = stablehlo.constant dense<[[(0.0, 0.0), (0.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a21:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %seed_21, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a21#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a21#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(-0.346154, 0.730769), (0.192308, 1.038462)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b21:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %dzero, %b, %seed_21) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b21#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b21#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.230769, -0.153846), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 21
  %rev_21:3 = enzyme.autodiff @triangular_solve_complex_left_lower(%A, %b, %seed_21) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_21#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_21#1, dense<[[(-0.923077, 0.884615), (0.0, 0.0)], [(-0.346154, -0.730769), (-0.464497, -0.014793)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_21#2, dense<[[(0.115385, -0.423077), (0.0, 0.0)], [(0.230769, 0.153846), (0.0, 0.0)]]> : tensor<2x2xcomplex<f32>>

  // complex fwd, ij = 22
  %seed_22 = stablehlo.constant dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f32>>
  %fwd_a22:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %seed_22, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_a22#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_a22#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(-0.464497, 0.014793), (-1.038462, 0.192308)]]> : tensor<2x2xcomplex<f32>>

  %fwd_b22:2 = enzyme.fwddiff @triangular_solve_complex_left_lower(%A, %dzero, %b, %seed_22) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b22#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b22#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.230769, -0.153846)]]> : tensor<2x2xcomplex<f32>>

  // complex rev, ij = 22
  %rev_22:3 = enzyme.autodiff @triangular_solve_complex_left_lower(%A, %b, %seed_22) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)

  check.expect_almost_eq %rev_22#0, %output : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_22#1, dense<[[(-1.653846, 0.230769), (0.0, 0.0)], [(0.192308, -1.038462), (-1.038462, -0.192308)]]> : tensor<2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev_22#2, dense<[[(0.0, 0.0), (0.115385, -0.423077)], [(0.0, 0.0), (0.230769, 0.153846)]]> : tensor<2x2xcomplex<f32>>

  return
}
