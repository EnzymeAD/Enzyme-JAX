// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=left_lower outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=left_lower outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=right_upper_adjoint_unit outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE-COMPLEX
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

// FORWARD:  func.func @left_lower(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x1xf32>, %arg3: tensor<2x1xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>) {
// FORWARD-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
// FORWARD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// FORWARD-NEXT:    %0 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
// FORWARD-NEXT:    %1 = stablehlo.select %c, %arg1, %cst : tensor<2x2xi1>, tensor<2x2xf32>
// FORWARD-NEXT:    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
// FORWARD-NEXT:    %3 = stablehlo.subtract %arg3, %2 : tensor<2x1xf32>
// FORWARD-NEXT:    %4 = "stablehlo.triangular_solve"(%arg0, %3) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
// FORWARD-NEXT:    return %0, %4 : tensor<2x1xf32>, tensor<2x1xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @left_lower(%arg0: tensor<2x2xf32>, %arg1: tensor<2x1xf32>, %arg2: tensor<2x1xf32>) -> (tensor<2x2xf32>, tensor<2x1xf32>) {
// REVERSE-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// REVERSE-NEXT:    %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
// REVERSE-NEXT:    %1 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
// REVERSE-NEXT:    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// REVERSE-NEXT:    %3 = stablehlo.negate %2 : tensor<2x2xf32>
// REVERSE-NEXT:    %4 = stablehlo.select %c, %3, %cst : tensor<2x2xi1>, tensor<2x2xf32>
// REVERSE-NEXT:    return %4, %1 : tensor<2x2xf32>, tensor<2x1xf32>
// REVERSE-NEXT:  }

// REVERSE-COMPLEX:  func.func @right_upper_adjoint_unit(%arg0: tensor<2x2x2xcomplex<f32>>, %arg1: tensor<2x1x2xcomplex<f32>>, %arg2: tensor<2x1x2xcomplex<f32>>) -> tensor<2x2x2xcomplex<f32>> {
// REVERSE-COMPLEX-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[[false, true], [false, false]], [[false, true], [false, false]]]> : tensor<2x2x2xi1>
// REVERSE-COMPLEX-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2x2x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose ADJOINT>, unit_diagonal = true}> : (tensor<2x2x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>) -> tensor<2x1x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %1 = "stablehlo.triangular_solve"(%arg0, %arg2) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<2x2x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>) -> tensor<2x1x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %2 = chlo.conj %1 : tensor<2x1x2xcomplex<f32>> -> tensor<2x1x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %3 = stablehlo.dot_general %2, %0, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<2x1x2xcomplex<f32>>, tensor<2x1x2xcomplex<f32>>) -> tensor<2x2x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %4 = stablehlo.negate %3 : tensor<2x2x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %5 = stablehlo.select %c, %4, %cst : tensor<2x2x2xi1>, tensor<2x2x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    return %5 : tensor<2x2x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:  }

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

  // width 2
  %wda3 = stablehlo.constant dense<[[[(-2.0, -1.0), (1.0, 1.0)], [(0.0, 1.0), (2.0, 1.0)]], [[(-1.0, -1.0), (-1.0, 0.0)], [(0.0, 1.0), (-1.0, 1.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %wdb3 = stablehlo.constant dense<[[[(-1.0, 0.0)], [(0.0, 0.0)]], [[(2.0, 1.0)], [(1.0, 1.0)]]]> : tensor<2x2x1xcomplex<f64>>
  %wg3 = stablehlo.constant dense<[[[(0.0, 1.0)], [(2.0, 1.0)]], [[(-2.0, -1.0)], [(1.0, -1.0)]]]> : tensor<2x2x1xcomplex<f64>>

  %wfwd3:2 = enzyme.fwddiff @left_upper_transpose_unit(%a3, %wda3, %b3, %wdb3) {
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>],
    width = 2 : i64
  } : (tensor<2x2xcomplex<f64>>, tensor<2x2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x2x1xcomplex<f64>>) -> (tensor<2x1xcomplex<f64>>, tensor<2x2x1xcomplex<f64>>)

  check.expect_almost_eq_const %wfwd3#1, dense<[[[(-1.0, 0.0)], [(-2.0, 1.0)]], [[(2.0, 1.0)], [(6.0, 1.0)]]]> : tensor<2x2x1xcomplex<f64>>

  %wrev3:2 = enzyme.autodiff @left_upper_transpose_unit(%a3, %b3, %wg3) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_activenoneed>],
    width = 2 : i64
  } : (tensor<2x2xcomplex<f64>>, tensor<2x1xcomplex<f64>>, tensor<2x2x1xcomplex<f64>>) -> (tensor<2x2x2xcomplex<f64>>, tensor<2x2x1xcomplex<f64>>)

  check.expect_almost_eq_const %wrev3#0, dense<[[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]], [[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]]> : tensor<2x2x2xcomplex<f64>>
  check.expect_almost_eq_const %wrev3#1, dense<[[[(3.0, 5.0)], [(2.0, 1.0)]], [[(1.0, -2.0)], [(1.0, -1.0)]]]> : tensor<2x2x1xcomplex<f64>>

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
