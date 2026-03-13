// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=tri_solve_lower_nt outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=tri_solve_lower_t outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-T

func.func @tri_solve_lower_nt(%arg0: tensor<3x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{
    left_side = true,
    lower = true,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>,
    unit_diagonal = false
  }> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}

// REVERSE-LABEL: func.func @tri_solve_lower_nt
// REVERSE-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x2xf64>, %[[XBAR:.+]]: tensor<3x2xf64>) -> (tensor<3x3xf64>, tensor<3x2xf64>) {
// REVERSE-NEXT:   %[[ZERO33:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-NEXT:   %[[ZERO32:.+]] = arith.constant dense<0.000000e+00> : tensor<3x2xf64>
// REVERSE-NEXT:   %[[ZERO33A:.+]] = arith.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-NEXT:   %[[X:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// REVERSE-NEXT:   %[[XBAR1:.+]] = arith.addf %[[XBAR]], %[[ZERO32]] : tensor<3x2xf64>
// REVERSE-NEXT:   %[[BBAR:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[XBAR1]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// REVERSE-NEXT:   %[[BBAR1:.+]] = arith.addf %[[BBAR]], %[[ZERO32]] : tensor<3x2xf64>
// REVERSE-NEXT:   %[[NEGBBAR:.+]] = stablehlo.negate %[[BBAR]] : tensor<3x2xf64>
// REVERSE-NEXT:   %[[XT:.+]] = stablehlo.transpose %[[X]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// REVERSE-NEXT:   %[[OUTER:.+]] = stablehlo.dot_general %[[NEGBBAR]], %[[XT]], contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[IOTA0:.+]] = stablehlo.iota dim = 0 : tensor<3x3xi64>
// REVERSE-NEXT:   %[[IOTA1:.+]] = stablehlo.iota dim = 1 : tensor<3x3xi64>
// REVERSE-NEXT:   %[[GE:.+]] = stablehlo.compare  GE, %[[IOTA0]], %[[IOTA1]] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
// REVERSE-NEXT:   %[[MASKED:.+]] = stablehlo.select %[[GE]], %[[OUTER]], %[[ZERO33]] : tensor<3x3xi1>, tensor<3x3xf64>
// REVERSE-NEXT:   %[[ABAR:.+]] = arith.addf %[[MASKED]], %[[ZERO33A]] : tensor<3x3xf64>
// REVERSE-NEXT:   return %[[ABAR]], %[[BBAR1]] : tensor<3x3xf64>, tensor<3x2xf64>

func.func @tri_solve_lower_t(%arg0: tensor<3x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{
    left_side = true,
    lower = true,
    transpose_a = #stablehlo<transpose TRANSPOSE>,
    unit_diagonal = false
  }> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}

// REVERSE-T-LABEL: func.func @tri_solve_lower_t
// REVERSE-T-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x2xf64>, %[[XBAR:.+]]: tensor<3x2xf64>) -> (tensor<3x3xf64>, tensor<3x2xf64>) {
// REVERSE-T-NEXT:   %[[ZERO33:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-T-NEXT:   %[[ZERO32:.+]] = arith.constant dense<0.000000e+00> : tensor<3x2xf64>
// REVERSE-T-NEXT:   %[[ZERO33A:.+]] = arith.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-T-NEXT:   %[[X:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[B]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// REVERSE-T-NEXT:   %[[XBAR1:.+]] = arith.addf %[[XBAR]], %[[ZERO32]] : tensor<3x2xf64>
// REVERSE-T-NEXT:   %[[BBAR:.+]] = "stablehlo.triangular_solve"(%[[A]], %[[XBAR1]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// REVERSE-T-NEXT:   %[[BBAR1:.+]] = arith.addf %[[BBAR]], %[[ZERO32]] : tensor<3x2xf64>
// REVERSE-T-NEXT:   %[[NEGX:.+]] = stablehlo.negate %[[X]] : tensor<3x2xf64>
// REVERSE-T-NEXT:   %[[BBART:.+]] = stablehlo.transpose %[[BBAR]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// REVERSE-T-NEXT:   %[[OUTER:.+]] = stablehlo.dot_general %[[NEGX]], %[[BBART]], contracting_dims = [1] x [0] : (tensor<3x2xf64>, tensor<2x3xf64>) -> tensor<3x3xf64>
// REVERSE-T-NEXT:   %[[IOTA0:.+]] = stablehlo.iota dim = 0 : tensor<3x3xi64>
// REVERSE-T-NEXT:   %[[IOTA1:.+]] = stablehlo.iota dim = 1 : tensor<3x3xi64>
// REVERSE-T-NEXT:   %[[GE:.+]] = stablehlo.compare  GE, %[[IOTA0]], %[[IOTA1]] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
// REVERSE-T-NEXT:   %[[MASKED:.+]] = stablehlo.select %[[GE]], %[[OUTER]], %[[ZERO33]] : tensor<3x3xi1>, tensor<3x3xf64>
// REVERSE-T-NEXT:   %[[ABAR:.+]] = arith.addf %[[MASKED]], %[[ZERO33A]] : tensor<3x3xf64>
// REVERSE-T-NEXT:   return %[[ABAR]], %[[BBAR1]] : tensor<3x3xf64>, tensor<3x2xf64>
