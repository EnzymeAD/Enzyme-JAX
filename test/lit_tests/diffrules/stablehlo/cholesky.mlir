// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_lower outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --canonicalize --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_lower outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cholesky_upper_batch outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE-UPPER

func.func @cholesky_lower(%a : tensor<3x3xf32>) -> tensor<3x3xf32> {
  %l = stablehlo.cholesky %a, lower = true : tensor<3x3xf32>
  func.return %l : tensor<3x3xf32>
}

func.func @cholesky_upper_batch(%a : tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
  %u = stablehlo.cholesky %a, lower = false : tensor<2x3x3xf64>
  func.return %u : tensor<2x3x3xf64>
}

// FORWARD:  func.func @cholesky_lower(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>) {
// FORWARD-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, false, false], [false, true, false], [false, false, true]]> : tensor<3x3xi1>
// FORWARD-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[false, false, false], [true, false, false], [true, true, false]]> : tensor<3x3xi1>
// FORWARD-NEXT{LITERAL}:    %c_1 = stablehlo.constant dense<[[true, false, false], [true, true, false], [true, true, true]]> : tensor<3x3xi1>
// FORWARD-NEXT:    %cst = stablehlo.constant dense<5.000000e-01> : tensor<3x3xf32>
// FORWARD-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// FORWARD-NEXT:    %0 = stablehlo.cholesky %arg0, lower = true : tensor<3x3xf32>
// FORWARD-NEXT:    %1 = stablehlo.select %c_1, %0, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %2 = stablehlo.multiply %arg1, %cst : tensor<3x3xf32>
// FORWARD-NEXT:    %3 = stablehlo.select %c, %2, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %4 = stablehlo.select %c_0, %arg1, %3 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %6 = stablehlo.add %4, %5 : tensor<3x3xf32>
// FORWARD-NEXT:    %7 = "stablehlo.triangular_solve"(%1, %6) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %8 = "stablehlo.triangular_solve"(%1, %7) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %9 = stablehlo.multiply %8, %cst : tensor<3x3xf32>
// FORWARD-NEXT:    %10 = stablehlo.select %c, %9, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %11 = stablehlo.select %c_0, %8, %10 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %12 = stablehlo.dot_general %1, %11, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    return %0, %12 : tensor<3x3xf32>, tensor<3x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @cholesky_lower(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
// REVERSE-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, false, false], [false, true, false], [false, false, true]]> : tensor<3x3xi1>
// REVERSE-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[false, false, false], [true, false, false], [true, true, false]]> : tensor<3x3xi1>
// REVERSE-NEXT{LITERAL}:    %c_1 = stablehlo.constant dense<[[true, false, false], [true, true, false], [true, true, true]]> : tensor<3x3xi1>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<5.000000e-01> : tensor<3x3xf32>
// REVERSE-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// REVERSE-NEXT:    %0 = stablehlo.cholesky %arg0, lower = true : tensor<3x3xf32>
// REVERSE-NEXT:    %1 = stablehlo.select %c_1, %0, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %2, %cst : tensor<3x3xf32>
// REVERSE-NEXT:    %4 = stablehlo.select %c, %3, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %5 = stablehlo.select %c_0, %2, %4 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %6 = "stablehlo.triangular_solve"(%1, %5) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %7 = "stablehlo.triangular_solve"(%1, %6) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %9 = stablehlo.add %7, %8 : tensor<3x3xf32>
// REVERSE-NEXT:    %10 = stablehlo.multiply %9, %cst : tensor<3x3xf32>
// REVERSE-NEXT:    %11 = stablehlo.select %c, %10, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %12 = stablehlo.select %c_0, %9, %11 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    return %12 : tensor<3x3xf32>
// REVERSE-NEXT:  }

// REVERSE-UPPER:  func.func @cholesky_upper_batch(%arg0: tensor<2x3x3xf64>, %arg1: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
// REVERSE-UPPER-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[[true, false, false], [false, true, false], [false, false, true]], [[true, false, false], [false, true, false], [false, false, true]]]> : tensor<2x3x3xi1>
// REVERSE-UPPER-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[[false, true, true], [false, false, true], [false, false, false]], [[false, true, true], [false, false, true], [false, false, false]]]> : tensor<2x3x3xi1>
// REVERSE-UPPER-NEXT{LITERAL}:    %c_1 = stablehlo.constant dense<[[[true, true, true], [false, true, true], [false, false, true]], [[true, true, true], [false, true, true], [false, false, true]]]> : tensor<2x3x3xi1>
// REVERSE-UPPER-NEXT:    %cst = stablehlo.constant dense<5.000000e-01> : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %0 = stablehlo.cholesky %arg0 : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %1 = stablehlo.select %c_1, %0, %cst_2 : tensor<2x3x3xi1>, tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %2 = stablehlo.dot_general %arg1, %1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [HIGHEST, HIGHEST] : (tensor<2x3x3xf64>, tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %3 = stablehlo.multiply %2, %cst : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %4 = stablehlo.select %c, %3, %cst_2 : tensor<2x3x3xi1>, tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %5 = stablehlo.select %c_0, %2, %4 : tensor<2x3x3xi1>, tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %6 = "stablehlo.triangular_solve"(%1, %5) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x3x3xf64>, tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %7 = "stablehlo.triangular_solve"(%1, %6) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x3x3xf64>, tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %8 = stablehlo.transpose %7, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %9 = stablehlo.add %7, %8 : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %10 = stablehlo.multiply %9, %cst : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %11 = stablehlo.select %c, %10, %cst_2 : tensor<2x3x3xi1>, tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    %12 = stablehlo.select %c_0, %9, %11 : tensor<2x3x3xi1>, tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:    return %12 : tensor<2x3x3xf64>
// REVERSE-UPPER-NEXT:  }
