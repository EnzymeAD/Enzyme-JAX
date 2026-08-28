// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_lower outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_upper outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-UPPER

func.func @cholesky_lower(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
  %0 = stablehlo.cholesky %arg0, lower = true : tensor<3x3xf64>
  return %0 : tensor<3x3xf64>
}

// REVERSE-LABEL: func.func @cholesky_lower
// REVERSE-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[LBAR:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
// REVERSE-NEXT:   %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<3x3xf64>
// REVERSE-NEXT:   %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-NEXT:   %[[ZERO2:.+]] = arith.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-NEXT:   %[[L:.+]] = stablehlo.cholesky %[[A]], lower = true : tensor<3x3xf64>
// REVERSE-NEXT:   %[[LBAR1:.+]] = arith.addf %[[LBAR]], %[[ZERO2]] : tensor<3x3xf64>
// REVERSE-NEXT:   %[[LT:.+]] = stablehlo.transpose %[[L]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[S:.+]] = stablehlo.dot_general %[[LT]], %[[LBAR1]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[IOTA0:.+]] = stablehlo.iota dim = 0 : tensor<3x3xi64>
// REVERSE-NEXT:   %[[IOTA1:.+]] = stablehlo.iota dim = 1 : tensor<3x3xi64>
// REVERSE-NEXT:   %[[GT:.+]] = stablehlo.compare  GT, %[[IOTA0]], %[[IOTA1]] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
// REVERSE-NEXT:   %[[EQ:.+]] = stablehlo.compare  EQ, %[[IOTA0]], %[[IOTA1]] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
// REVERSE-NEXT:   %[[STRICT:.+]] = stablehlo.select %[[GT]], %[[S]], %[[ZERO]] : tensor<3x3xi1>, tensor<3x3xf64>
// REVERSE-NEXT:   %[[DHALF:.+]] = stablehlo.multiply %[[S]], %[[HALF]] : tensor<3x3xf64>
// REVERSE-NEXT:   %[[DIAG:.+]] = stablehlo.select %[[EQ]], %[[DHALF]], %[[ZERO]] : tensor<3x3xi1>, tensor<3x3xf64>
// REVERSE-NEXT:   %[[PHI:.+]] = stablehlo.add %[[STRICT]], %[[DIAG]] : tensor<3x3xf64>
// REVERSE-NEXT:   %[[Y:.+]] = "stablehlo.triangular_solve"(%[[L]], %[[PHI]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[YT:.+]] = stablehlo.transpose %[[Y]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[W:.+]] = "stablehlo.triangular_solve"(%[[L]], %[[YT]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[WT:.+]] = stablehlo.transpose %[[W]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[WTT:.+]] = stablehlo.transpose %[[WT]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-NEXT:   %[[SUM:.+]] = stablehlo.add %[[WT]], %[[WTT]] : tensor<3x3xf64>
// REVERSE-NEXT:   %[[SYM:.+]] = stablehlo.multiply %[[SUM]], %[[HALF]] : tensor<3x3xf64>
// REVERSE-NEXT:   %[[RES:.+]] = arith.addf %[[SYM]], %[[ZERO2]] : tensor<3x3xf64>
// REVERSE-NEXT:   return %[[RES]] : tensor<3x3xf64>

func.func @cholesky_upper(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
  %0 = stablehlo.cholesky %arg0 : tensor<3x3xf64>
  return %0 : tensor<3x3xf64>
}

// REVERSE-UPPER-LABEL: func.func @cholesky_upper
// REVERSE-UPPER-SAME: (%[[A:.+]]: tensor<3x3xf64>, %[[UBAR:.+]]: tensor<3x3xf64>) -> tensor<3x3xf64> {
// REVERSE-UPPER-NEXT:   %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[ZERO2:.+]] = arith.constant dense<0.000000e+00> : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[U:.+]] = stablehlo.cholesky %[[A]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[UBAR1:.+]] = arith.addf %[[UBAR]], %[[ZERO2]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[UBART:.+]] = stablehlo.transpose %[[UBAR1]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[S:.+]] = stablehlo.dot_general %[[U]], %[[UBART]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[IOTA0:.+]] = stablehlo.iota dim = 0 : tensor<3x3xi64>
// REVERSE-UPPER-NEXT:   %[[IOTA1:.+]] = stablehlo.iota dim = 1 : tensor<3x3xi64>
// REVERSE-UPPER-NEXT:   %[[GT:.+]] = stablehlo.compare  GT, %[[IOTA0]], %[[IOTA1]] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
// REVERSE-UPPER-NEXT:   %[[EQ:.+]] = stablehlo.compare  EQ, %[[IOTA0]], %[[IOTA1]] : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
// REVERSE-UPPER-NEXT:   %[[STRICT:.+]] = stablehlo.select %[[GT]], %[[S]], %[[ZERO]] : tensor<3x3xi1>, tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[DHALF:.+]] = stablehlo.multiply %[[S]], %[[HALF]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[DIAG:.+]] = stablehlo.select %[[EQ]], %[[DHALF]], %[[ZERO]] : tensor<3x3xi1>, tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[PHI:.+]] = stablehlo.add %[[STRICT]], %[[DIAG]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[Y:.+]] = "stablehlo.triangular_solve"(%[[U]], %[[PHI]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[YT:.+]] = stablehlo.transpose %[[Y]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[W:.+]] = "stablehlo.triangular_solve"(%[[U]], %[[YT]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[WT:.+]] = stablehlo.transpose %[[W]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[WTT:.+]] = stablehlo.transpose %[[WT]], dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[SUM:.+]] = stablehlo.add %[[WT]], %[[WTT]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[SYM:.+]] = stablehlo.multiply %[[SUM]], %[[HALF]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   %[[RES:.+]] = arith.addf %[[SYM]], %[[ZERO2]] : tensor<3x3xf64>
// REVERSE-UPPER-NEXT:   return %[[RES]] : tensor<3x3xf64>
