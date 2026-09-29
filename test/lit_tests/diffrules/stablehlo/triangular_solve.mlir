// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @left_lower(%a : tensor<2x2xf32>, %b : tensor<2x1xf32>) -> tensor<2x1xf32> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  func.return %x : tensor<2x1xf32>
}

func.func @left_lower_unit(%a : tensor<2x2xcomplex<f64>>, %b : tensor<2x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
  func.return %x : tensor<2x1xcomplex<f64>>
}

func.func @left_upper_transpose_unit(%a : tensor<2x2xcomplex<f64>>, %b : tensor<2x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = true, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
  func.return %x : tensor<2x1xcomplex<f64>>
}

func.func @left_lower_adjoint(%a : tensor<2x2xcomplex<f64>>, %b : tensor<2x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose ADJOINT>} : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
  func.return %x : tensor<2x1xcomplex<f64>>
}

func.func @right_upper(%a : tensor<2x2xcomplex<f64>>, %b : tensor<1x2xcomplex<f64>>) -> tensor<1x2xcomplex<f64>> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>) -> tensor<1x2xcomplex<f64>>
  func.return %x : tensor<1x2xcomplex<f64>>
}

func.func @right_lower_transpose(%a : tensor<2x2xcomplex<f64>>, %b : tensor<1x2xcomplex<f64>>) -> tensor<1x2xcomplex<f64>> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>) -> tensor<1x2xcomplex<f64>>
  func.return %x : tensor<1x2xcomplex<f64>>
}

func.func @right_upper_adjoint_unit(%a : tensor<2x2x2xcomplex<f32>>, %b : tensor<2x1x2xcomplex<f32>>) -> tensor<2x1x2xcomplex<f32>> {
  %x = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = true, transpose_a = #stablehlo<transpose ADJOINT>} : (tensor<2x2x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>) -> tensor<2x1x2xcomplex<f32>>
  func.return %x : tensor<2x1x2xcomplex<f32>>
}

func.func @main() {
  // left lower
  %a1 = stablehlo.constant dense<[[-1.0, 1.0], [-1.0, 1.0]]> : tensor<2x2xf32>
  %b1 = stablehlo.constant dense<[[1.0], [1.0]]> : tensor<2x1xf32>
  %da1 = stablehlo.constant dense<[[1.0, 0.0], [-2.0, 0.0]]> : tensor<2x2xf32>
  %db1 = stablehlo.constant dense<[[-1.0], [-2.0]]> : tensor<2x1xf32>
  %g1 = stablehlo.constant dense<[[0.0], [1.0]]> : tensor<2x1xf32>

  %fwd1:2 = enzyme.fwddiff @left_lower(%a1, %da1, %b1, %db1) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>)

  check.expect_almost_eq_const %fwd1#0, dense<[[-1.0], [0.0]]> : tensor<2x1xf32>
  check.expect_almost_eq_const %fwd1#1, dense<[[0.0], [-4.0]]> : tensor<2x1xf32>

  %rev1:3 = enzyme.autodiff @left_lower(%a1, %b1, %g1) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> (tensor<2x1xf32>, tensor<2x2xf32>, tensor<2x1xf32>)

  check.expect_almost_eq_const %rev1#1, dense<[[-1.0, 0.0], [1.0, 0.0]]> : tensor<2x2xf32>
  check.expect_almost_eq_const %rev1#2, dense<[[-1.0], [1.0]]> : tensor<2x1xf32>

  // left lower unit
  %a2 = stablehlo.constant dense<[[(7.0, 0.0), (-1.0, -1.0)], [(-2.0, 0.0), (7.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %b2 = stablehlo.constant dense<[[(0.0, 0.0)], [(2.0, 1.0)]]> : tensor<2x1xcomplex<f64>>
  %da2 = stablehlo.constant dense<[[(2.0, 1.0), (0.0, 0.0)], [(1.0, 0.0), (0.0, 1.0)]]> : tensor<2x2xcomplex<f64>>
  %db2 = stablehlo.constant dense<[[(-2.0, -1.0)], [(0.0, 1.0)]]> : tensor<2x1xcomplex<f64>>
  %g2 = stablehlo.constant dense<[[(1.0, 1.0)], [(0.0, 0.0)]]> : tensor<2x1xcomplex<f64>>

  %fwd2:2 = enzyme.fwddiff @left_lower_unit(%a2, %da2, %b2, %db2) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>)

  check.expect_almost_eq_const %fwd2#0, dense<[[(0.0, 0.0)], [(2.0, 1.0)]]> : tensor<2x1xcomplex<f64>>
  check.expect_almost_eq_const %fwd2#1, dense<[[(-2.0, -1.0)], [(-4.0, -1.0)]]> : tensor<2x1xcomplex<f64>>

  %rev2:3 = enzyme.autodiff @left_lower_unit(%a2, %b2, %g2) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>)

  check.expect_almost_eq_const %rev2#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %rev2#2, dense<[[(1.0, 1.0)], [(0.0, 0.0)]]> : tensor<2x1xcomplex<f64>>

  // left upper transpose unit
  %a3 = stablehlo.constant dense<[[(7.0, 0.0), (-2.0, 1.0)], [(-2.0, 0.0), (7.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %b3 = stablehlo.constant dense<[[(0.0, 0.0)], [(-2.0, 1.0)]]> : tensor<2x1xcomplex<f64>>
  %da3 = stablehlo.constant dense<[[(1.0, 1.0), (2.0, -1.0)], [(1.0, 0.0), (2.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %db3 = stablehlo.constant dense<[[(1.0, -1.0)], [(1.0, -1.0)]]> : tensor<2x1xcomplex<f64>>
  %g3 = stablehlo.constant dense<[[(-1.0, 1.0)], [(2.0, 1.0)]]> : tensor<2x1xcomplex<f64>>

  %fwd3:2 = enzyme.fwddiff @left_upper_transpose_unit(%a3, %da3, %b3, %db3) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>)

  check.expect_almost_eq_const %fwd3#0, dense<[[(0.0, 0.0)], [(-2.0, 1.0)]]> : tensor<2x1xcomplex<f64>>
  check.expect_almost_eq_const %fwd3#1, dense<[[(1.0, -1.0)], [(2.0, -4.0)]]> : tensor<2x1xcomplex<f64>>

  %rev3:3 = enzyme.autodiff @left_upper_transpose_unit(%a3, %b3, %g3) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>)

  check.expect_almost_eq_const %rev3#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %rev3#2, dense<[[(2.0, 5.0)], [(2.0, 1.0)]]> : tensor<2x1xcomplex<f64>>

  // left lower adjoint
  %a4 = stablehlo.constant dense<[[(-1.0, 0.0), (2.0, -1.0)], [(1.0, -1.0), (2.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %b4 = stablehlo.constant dense<[[(1.0, -1.0)], [(-2.0, 0.0)]]> : tensor<2x1xcomplex<f64>>
  %da4 = stablehlo.constant dense<[[(-1.0, -1.0), (2.0, 1.0)], [(-1.0, 1.0), (2.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %db4 = stablehlo.constant dense<[[(1.0, 1.0)], [(-1.0, -1.0)]]> : tensor<2x1xcomplex<f64>>
  %g4 = stablehlo.constant dense<[[(2.0, 1.0)], [(-2.0, 0.0)]]> : tensor<2x1xcomplex<f64>>

  %fwd4:2 = enzyme.fwddiff @left_lower_adjoint(%a4, %da4, %b4, %db4) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>)

  check.expect_almost_eq_const %fwd4#0, dense<[[(-2.0, 0.0)], [(-1.0, 0.0)]]> : tensor<2x1xcomplex<f64>>
  check.expect_almost_eq_const %fwd4#1, dense<[[(3.0, -2.0)], [(0.5, -0.5)]]> : tensor<2x1xcomplex<f64>>

  %rev4:3 = enzyme.autodiff @left_lower_adjoint(%a4, %b4, %g4) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>)

  check.expect_almost_eq_const %rev4#1, dense<[[(-4.0, 2.0), (0.0, 0.0)], [(-2.0, 1.0), (0.5, 0.5)]]> : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %rev4#2, dense<[[(-2.0, -1.0)], [(0.5, -0.5)]]> : tensor<2x1xcomplex<f64>>

  // right upper
  %a5 = stablehlo.constant dense<[[(-1.0, 0.0), (-2.0, 0.0)], [(0.0, -1.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %b5 = stablehlo.constant dense<[[(-2.0, 0.0), (1.0, 0.0)]]> : tensor<1x2xcomplex<f64>>
  %da5 = stablehlo.constant dense<[[(1.0, 1.0), (2.0, 0.0)], [(-2.0, -1.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %db5 = stablehlo.constant dense<[[(-2.0, 1.0), (-1.0, 1.0)]]> : tensor<1x2xcomplex<f64>>
  %g5 = stablehlo.constant dense<[[(-1.0, 1.0), (2.0, 0.0)]]> : tensor<1x2xcomplex<f64>>

  %fwd5:2 = enzyme.fwddiff @right_upper(%a5, %da5, %b5, %db5) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>) -> (tensor<1x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>)

  check.expect_almost_eq_const %fwd5#0, dense<[[(2.0, 0.0), (5.0, 0.0)]]> : tensor<1x2xcomplex<f64>>
  check.expect_almost_eq_const %fwd5#1, dense<[[(4.0, 1.0), (-2.0, 3.0)]]> : tensor<1x2xcomplex<f64>>

  %rev5:3 = enzyme.autodiff @right_upper(%a5, %b5, %g5) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>) -> (tensor<1x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>)

  check.expect_almost_eq_const %rev5#1, dense<[[(6.0, 2.0), (-4.0, 0.0)], [(0.0, 0.0), (-10.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %rev5#2, dense<[[(-3.0, -1.0), (2.0, 0.0)]]> : tensor<1x2xcomplex<f64>>

  // right lower transpose
  %a6 = stablehlo.constant dense<[[(1.0, 0.0), (-2.0, 0.0)], [(1.0, 0.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %b6 = stablehlo.constant dense<[[(0.0, 0.0), (2.0, 1.0)]]> : tensor<1x2xcomplex<f64>>
  %da6 = stablehlo.constant dense<[[(-2.0, 1.0), (0.0, 0.0)], [(-1.0, 1.0), (1.0, -1.0)]]> : tensor<2x2xcomplex<f64>>
  %db6 = stablehlo.constant dense<[[(1.0, -1.0), (0.0, 1.0)]]> : tensor<1x2xcomplex<f64>>
  %g6 = stablehlo.constant dense<[[(2.0, 1.0), (-2.0, 1.0)]]> : tensor<1x2xcomplex<f64>>

  %fwd6:2 = enzyme.fwddiff @right_lower_transpose(%a6, %da6, %b6, %db6) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>) -> (tensor<1x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>)

  check.expect_almost_eq_const %fwd6#0, dense<[[(0.0, 0.0), (2.0, 1.0)]]> : tensor<1x2xcomplex<f64>>
  check.expect_almost_eq_const %fwd6#1, dense<[[(1.0, -1.0), (-4.0, 3.0)]]> : tensor<1x2xcomplex<f64>>

  %rev6:3 = enzyme.autodiff @right_lower_transpose(%a6, %b6, %g6) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>) -> (tensor<1x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<1x2xcomplex<f64>>)

  check.expect_almost_eq_const %rev6#1, dense<[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (3.0, -4.0)]]> : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %rev6#2, dense<[[(4.0, 0.0), (-2.0, 1.0)]]> : tensor<1x2xcomplex<f64>>

  // right upper adjoint unit
  %a7 = stablehlo.constant dense<[[[(7.0, 0.0), (0.0, 1.0)], [(2.0, 0.0), (7.0, 0.0)]], [[(7.0, 0.0), (-1.0, -1.0)], [(-1.0, 0.0), (7.0, 0.0)]]]> : tensor<2x2x2xcomplex<f32>>
  %b7 = stablehlo.constant dense<[[[(2.0, 1.0), (1.0, 1.0)]], [[(1.0, 0.0), (0.0, 0.0)]]]> : tensor<2x1x2xcomplex<f32>>
  %da7 = stablehlo.constant dense<[[[(-1.0, 1.0), (0.0, 1.0)], [(1.0, 0.0), (2.0, 1.0)]], [[(1.0, 0.0), (0.0, 1.0)], [(2.0, 1.0), (2.0, 1.0)]]]> : tensor<2x2x2xcomplex<f32>>
  %db7 = stablehlo.constant dense<[[[(0.0, 0.0), (0.0, 1.0)]], [[(1.0, -1.0), (1.0, 1.0)]]]> : tensor<2x1x2xcomplex<f32>>
  %g7 = stablehlo.constant dense<[[[(0.0, -1.0), (0.0, -1.0)]], [[(2.0, 0.0), (2.0, -1.0)]]]> : tensor<2x1x2xcomplex<f32>>

  %fwd7:2 = enzyme.fwddiff @right_upper_adjoint_unit(%a7, %da7, %b7, %db7) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  } : (tensor<2x2x2xcomplex<f32>>, tensor<2x2x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>) -> (tensor<2x1x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>)

  check.expect_almost_eq_const %fwd7#0, dense<[[[(1.0, 2.0), (1.0, 1.0)]], [[(1.0, 0.0), (0.0, 0.0)]]]> : tensor<2x1x2xcomplex<f32>>
  check.expect_almost_eq_const %fwd7#1, dense<[[[(-2.0, 1.0), (0.0, 1.0)]], [[(3.0, -1.0), (1.0, 1.0)]]]> : tensor<2x1x2xcomplex<f32>>

  %rev7:3 = enzyme.autodiff @right_upper_adjoint_unit(%a7, %b7, %g7) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2x2x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>) -> (tensor<2x1x2xcomplex<f32>>, tensor<2x2x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>)

  check.expect_almost_eq_const %rev7#1, dense<[[[(0.0, 0.0), (1.0, -1.0)], [(0.0, 0.0), (0.0, 0.0)]], [[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]]> : tensor<2x2x2xcomplex<f32>>
  check.expect_almost_eq_const %rev7#2, dense<[[[(0.0, -1.0), (-1.0, -1.0)]], [[(2.0, 0.0), (4.0, 1.0)]]]> : tensor<2x1x2xcomplex<f32>>

  func.return
}
