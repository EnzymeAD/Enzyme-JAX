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

// Reverse-mode AD for triangular solve (AX = B, left_side=true, no transpose):
//   B_bar = solve(A^T, X_bar)
//   A_bar = -B_bar @ X^T, masked to lower triangle
//
// REVERSE-LABEL: func.func @tri_solve_lower_nt
// REVERSE-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x2xf64>, %[[XBAR:.+]]: tensor<3x2xf64>) -> (tensor<3x3xf64>, tensor<3x2xf64>)
// Forward solve (cached for X)
// REVERSE: "stablehlo.triangular_solve"(%[[A]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// Adjoint solve: B_bar = solve(A^T, X_bar) with flipped transpose
// REVERSE: "stablehlo.triangular_solve"(%[[A]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// A_bar = -B_bar @ X^T
// REVERSE: stablehlo.negate
// REVERSE: stablehlo.dot_general {{.+}} contracting_dims = [1] x [0]
// Triangle masking (lower, GE = including diagonal since unit_diagonal=false)
// REVERSE: stablehlo.iota dim = 0 : tensor<3x3xi64>
// REVERSE: stablehlo.iota dim = 1 : tensor<3x3xi64>
// REVERSE: stablehlo.compare GE
// REVERSE: stablehlo.select
// REVERSE: return %{{.+}}, %{{.+}} : tensor<3x3xf64>, tensor<3x2xf64>

func.func @tri_solve_lower_t(%arg0: tensor<3x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{
    left_side = true,
    lower = true,
    transpose_a = #stablehlo<transpose TRANSPOSE>,
    unit_diagonal = false
  }> : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}

// Reverse-mode AD for A^T X = B (left_side=true, transpose):
//   B_bar = solve(A, X_bar)   [flipped: TRANSPOSE -> NO_TRANSPOSE]
//   A_bar = -X @ B_bar^T, masked to lower triangle
//
// REVERSE-T-LABEL: func.func @tri_solve_lower_t
// REVERSE-T-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x2xf64>, %[[XBAR:.+]]: tensor<3x2xf64>) -> (tensor<3x3xf64>, tensor<3x2xf64>)
// Forward solve (with TRANSPOSE)
// REVERSE-T: "stablehlo.triangular_solve"(%[[A]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}>
// Adjoint solve: B_bar = solve(A, X_bar) with NO_TRANSPOSE
// REVERSE-T: "stablehlo.triangular_solve"(%[[A]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}>
// A_bar = -X @ B_bar^T (note: for TRANSPOSE case, formula differs)
// REVERSE-T: stablehlo.negate
// REVERSE-T: stablehlo.dot_general
// Triangle masking
// REVERSE-T: stablehlo.iota
// REVERSE-T: stablehlo.iota
// REVERSE-T: stablehlo.compare GE
// REVERSE-T: stablehlo.select
// REVERSE-T: return %{{.+}}, %{{.+}} : tensor<3x3xf64>, tensor<3x2xf64>
