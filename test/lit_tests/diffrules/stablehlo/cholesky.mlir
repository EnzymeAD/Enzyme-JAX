// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=upper outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=UPPER
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lower outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=LOWER

// Below 20 rows the reverse rule is a single block.

func.func @upper(%a : tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.cholesky %a, lower = false : tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

func.func @lower(%a : tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.cholesky %a, lower = true : tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// UPPER-LABEL: func.func @upper(
// UPPER-SAME:      %[[A:.+]]: tensor<4x4xf64>, %[[UBAR:.+]]: tensor<4x4xf64>) -> tensor<4x4xf64> {
// UPPER-DAG:     %[[QUARTER:.+]] = stablehlo.constant dense<2.500000e-01> : tensor<4x4xf64>
// UPPER-DAG:     %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
// UPPER-DAG:     %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf64>
// UPPER:         %[[U:.+]] = stablehlo.cholesky %[[A]] : tensor<4x4xf64>
// UPPER-NEXT:    %[[R0:.+]] = stablehlo.iota dim = 0 : tensor<4x4xi64>
// UPPER-NEXT:    %[[C0:.+]] = stablehlo.iota dim = 1 : tensor<4x4xi64>
// UPPER-NEXT:    %[[DIAG:.+]] = stablehlo.compare EQ, %[[R0]], %[[C0]] : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
// UPPER-NEXT:    %[[EYE:.+]] = stablehlo.select %[[DIAG]], %[[ONE]], %[[ZERO]] : tensor<4x4xi1>, tensor<4x4xf64>
// UPPER-NEXT:    %[[DINVT:.+]] = "stablehlo.triangular_solve"(%[[U]], %[[EYE]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// UPPER-NEXT:    %[[S:.+]] = stablehlo.dot_general %[[U]], %[[UBAR]], contracting_dims = [1] x [1] : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// UPPER-NEXT:    %[[R1:.+]] = stablehlo.iota dim = 0 : tensor<4x4xi64>
// UPPER-NEXT:    %[[C1:.+]] = stablehlo.iota dim = 1 : tensor<4x4xi64>
// UPPER-NEXT:    %[[LOW:.+]] = stablehlo.compare GE, %[[R1]], %[[C1]] : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
// UPPER-NEXT:    %[[ST:.+]] = stablehlo.transpose %[[S]], dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
// UPPER-NEXT:    %[[P:.+]] = stablehlo.select %[[LOW]], %[[S]], %[[ST]] : tensor<4x4xi1>, tensor<4x4xf64>
// UPPER-NEXT:    %[[W:.+]] = stablehlo.dot_general %[[DINVT]], %[[P]], contracting_dims = [1] x [0] : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// UPPER-NEXT:    %[[Z:.+]] = stablehlo.dot_general %[[W]], %[[DINVT]], contracting_dims = [1] x [1] : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// UPPER-NEXT:    %[[ZT:.+]] = stablehlo.transpose %[[Z]], dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
// UPPER-NEXT:    %[[SUM:.+]] = stablehlo.add %[[Z]], %[[ZT]] : tensor<4x4xf64>
// UPPER-NEXT:    %[[ABAR:.+]] = stablehlo.multiply %[[SUM]], %[[QUARTER]] : tensor<4x4xf64>
// UPPER-NEXT:    return %[[ABAR]] : tensor<4x4xf64>

// LOWER-LABEL: func.func @lower(
// LOWER-SAME:      %[[A:.+]]: tensor<4x4xf64>, %[[LBAR:.+]]: tensor<4x4xf64>) -> tensor<4x4xf64> {
// LOWER:         %[[L:.+]] = stablehlo.cholesky %[[A]], lower = true : tensor<4x4xf64>
// LOWER-NEXT:    %{{.+}} = stablehlo.iota dim = 0 : tensor<4x4xi64>
// LOWER-NEXT:    %{{.+}} = stablehlo.iota dim = 1 : tensor<4x4xi64>
// LOWER-NEXT:    %{{.+}} = stablehlo.compare EQ, %{{.+}}, %{{.+}} : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
// LOWER-NEXT:    %[[EYE:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<4x4xi1>, tensor<4x4xf64>
// LOWER-NEXT:    %[[DINVT:.+]] = "stablehlo.triangular_solve"(%[[L]], %[[EYE]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// LOWER-NEXT:    %[[S:.+]] = stablehlo.dot_general %[[L]], %[[LBAR]], contracting_dims = [0] x [0] : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// LOWER:         %[[P:.+]] = stablehlo.select %{{.+}}, %[[S]], %{{.+}} : tensor<4x4xi1>, tensor<4x4xf64>
// LOWER-NEXT:    %[[W:.+]] = stablehlo.dot_general %[[DINVT]], %[[P]], contracting_dims = [1] x [0] : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
// LOWER-NEXT:    %[[Z:.+]] = stablehlo.dot_general %[[W]], %[[DINVT]], contracting_dims = [1] x [1] : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
