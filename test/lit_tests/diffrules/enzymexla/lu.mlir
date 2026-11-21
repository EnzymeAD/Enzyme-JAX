// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lu outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --canonicalize --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lu outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE

func.func @lu(%a : tensor<3x3xf32>) -> tensor<3x3xf32> {
  %lu:4 = enzymexla.linalg.lu %a : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>)
  func.return %lu#0 : tensor<3x3xf32>
}

// FORWARD:  func.func @lu(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>) {
// FORWARD-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, true, true], [false, true, true], [false, false, true]]> : tensor<3x3xi1>
// FORWARD-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[false, false, false], [true, false, false], [true, true, false]]> : tensor<3x3xi1>
// FORWARD-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf32>
// FORWARD-NEXT{LITERAL}:    %c_1 = stablehlo.constant dense<[[1, 2, 3], [1, 2, 3], [1, 2, 3]]> : tensor<3x3xi32>
// FORWARD-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// FORWARD-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<3x3xf32>
// FORWARD-NEXT:    %output, %pivots, %permutation, %info = enzymexla.linalg.lu %arg0 {enzymexla.non_negative = [#enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>, #enzymexla.guaranteed<UNKNOWN>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>)
// FORWARD-NEXT:    %0 = stablehlo.broadcast_in_dim %permutation, dims = [0] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3xi32>) -> tensor<3x3xi32>
// FORWARD-NEXT:    %1 = stablehlo.compare EQ, %0, %c_1 : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
// FORWARD-NEXT:    %2 = stablehlo.select %1, %cst_3, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %3 = stablehlo.dot_general %2, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %4 = "stablehlo.triangular_solve"(%output, %3) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %5 = "stablehlo.triangular_solve"(%output, %4) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %6 = stablehlo.select %c_0, %output, %cst : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %7 = stablehlo.select %c_0, %5, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %8 = stablehlo.dot_general %6, %7, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %9 = stablehlo.select %c, %5, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %10 = stablehlo.select %c, %output, %cst_2 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %11 = stablehlo.dot_general %9, %10, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %12 = stablehlo.add %8, %11 : tensor<3x3xf32>
// FORWARD-NEXT:    return %output, %12 : tensor<3x3xf32>, tensor<3x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @lu(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
// REVERSE-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[1, 2, 3], [1, 2, 3], [1, 2, 3]]> : tensor<3x3xi32>
// REVERSE-NEXT{LITERAL}:    %c_0 = stablehlo.constant dense<[[true, true, true], [false, true, true], [false, false, true]]> : tensor<3x3xi1>
// REVERSE-NEXT{LITERAL}:    %c_1 = stablehlo.constant dense<[[false, false, false], [true, false, false], [true, true, false]]> : tensor<3x3xi1>
// REVERSE-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf32>
// REVERSE-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3x3xf32>
// REVERSE-NEXT:    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// REVERSE-NEXT:    %output, %pivots, %permutation, %info = enzymexla.linalg.lu %arg0 {enzymexla.non_negative = [#enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>, #enzymexla.guaranteed<UNKNOWN>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>)
// REVERSE-NEXT:    %0 = stablehlo.select %c_1, %output, %cst : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %2 = stablehlo.select %c_0, %output, %cst_3 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %3 = stablehlo.dot_general %arg1, %2, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %4 = stablehlo.select %c_1, %1, %3 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %5 = "stablehlo.triangular_solve"(%output, %4) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = true}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %6 = "stablehlo.triangular_solve"(%output, %5) <{left_side = false, lower = false, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %7 = stablehlo.broadcast_in_dim %permutation, dims = [0] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3xi32>) -> tensor<3x3xi32>
// REVERSE-NEXT:    %8 = stablehlo.compare EQ, %7, %c : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
// REVERSE-NEXT:    %9 = stablehlo.select %8, %cst_2, %cst_3 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %10 = stablehlo.dot_general %9, %6, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    return %10 : tensor<3x3xf32>
// REVERSE-NEXT:  }
