// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_lower outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_upper outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-UPPER

func.func @cholesky_lower(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
  %0 = stablehlo.cholesky %arg0, lower = true : tensor<3x3xf64>
  return %0 : tensor<3x3xf64>
}

// Reverse-mode AD for Cholesky uses Murray (2016):
//   S = L^T @ L_bar, Phi = tril(S) with halved diag,
//   A_bar = sym(L^{-T} Phi L^{-1})
//
// REVERSE-LABEL: func.func @cholesky_lower
// REVERSE-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[LBAR:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64>
// Forward Cholesky (cached for reverse)
// REVERSE: %[[L:.+]] = stablehlo.cholesky %[[A]], lower = true : tensor<3x3xf64>
// S = L^T @ L_bar
// REVERSE: stablehlo.transpose %[[L]], dims = [1, 0]
// REVERSE: stablehlo.dot_general {{.+}} contracting_dims = [1] x [0]
// Phi: triangle extraction with halved diagonal
// REVERSE: stablehlo.iota dim = 0 : tensor<3x3xi64>
// REVERSE: stablehlo.iota dim = 1 : tensor<3x3xi64>
// REVERSE: stablehlo.compare GT
// REVERSE: stablehlo.compare EQ
// Two triangular solves: Y = L^{-T} Phi, Z = L^{-T} Y^T
// REVERSE: "stablehlo.triangular_solve"(%[[L]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE: "stablehlo.triangular_solve"(%[[L]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// Return A_bar (symmetrized)
// REVERSE: return %{{.+}} : tensor<3x3xf64>

func.func @cholesky_upper(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
  %0 = stablehlo.cholesky %arg0 : tensor<3x3xf64>
  return %0 : tensor<3x3xf64>
}

// Reverse-mode for upper Cholesky (A = U^T U):
//   S = U @ U_bar^T, Phi = tril(S) with halved diag,
//   A_bar = sym(U^{-1} Phi U^{-T})
//
// REVERSE-UPPER-LABEL: func.func @cholesky_upper
// REVERSE-UPPER-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[UBAR:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64>
// Forward upper Cholesky (cached)
// REVERSE-UPPER: %[[U:.+]] = stablehlo.cholesky %[[A]] : tensor<3x3xf64>
// S = U @ U_bar^T
// REVERSE-UPPER: stablehlo.dot_general
// Phi: triangle extraction
// REVERSE-UPPER: stablehlo.iota
// REVERSE-UPPER: stablehlo.iota
// REVERSE-UPPER: stablehlo.compare GT
// REVERSE-UPPER: stablehlo.compare EQ
// Two triangular solves with upper triangle: Y = U^{-1} Phi, Z = U^{-1} Y^T
// REVERSE-UPPER: "stablehlo.triangular_solve"(%[[U]], %{{.+}}) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER: "stablehlo.triangular_solve"(%[[U]], %{{.+}}) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER: return %{{.+}} : tensor<3x3xf64>
