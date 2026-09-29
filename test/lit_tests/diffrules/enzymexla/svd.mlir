// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=svdvals outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE-VALUES
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=svd outfn= retTys=enzyme_dup,enzyme_dup,enzyme_dup argTys=enzyme_dup mode=ForwardMode" --canonicalize --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=svd outfn= retTys=enzyme_active,enzyme_active,enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE

func.func @svdvals(%a : tensor<4x3xf32>) -> tensor<3xf32> {
  %s:4 = enzymexla.linalg.svd %a : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<i32>)
  func.return %s#1 : tensor<3xf32>
}

func.func @svd(%a : tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>) {
  %s:4 = enzymexla.linalg.svd %a : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<i32>)
  func.return %s#0, %s#1, %s#2 : tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>
}

// REVERSE-VALUES:  func.func @svdvals(%arg0: tensor<4x3xf32>, %arg1: tensor<3xf32>) -> tensor<4x3xf32> {
// REVERSE-VALUES-NEXT:    %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<i32>)
// REVERSE-VALUES-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf32>) -> tensor<4x3xf32>
// REVERSE-VALUES-NEXT:    %1 = stablehlo.multiply %U, %0 : tensor<4x3xf32>
// REVERSE-VALUES-NEXT:    %2 = stablehlo.dot_general %1, %Vt, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<4x3xf32>, tensor<3x3xf32>) -> tensor<4x3xf32>
// REVERSE-VALUES-NEXT:    return %2 : tensor<4x3xf32>
// REVERSE-VALUES-NEXT:  }

// FORWARD:  func.func @svd(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) {
// FORWARD-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, false, false], [false, true, false], [false, false, true]]> : tensor<3x3xi1>
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x3xf32>
// FORWARD-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// FORWARD-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// FORWARD-NEXT:    %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<i32>)
// FORWARD-NEXT:    %0 = stablehlo.dot_general %U, %arg1, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %1 = stablehlo.dot_general %0, %Vt, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %2 = stablehlo.select %c, %1, %cst_1 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %3 = stablehlo.reduce(%2 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
// FORWARD-NEXT:    %4 = stablehlo.multiply %S, %S : tensor<3xf32>
// FORWARD-NEXT:    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %6 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %7 = stablehlo.subtract %5, %6 : tensor<3x3xf32>
// FORWARD-NEXT:    %8 = stablehlo.select %c, %cst, %7 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %9 = stablehlo.divide %cst, %8 : tensor<3x3xf32>
// FORWARD-NEXT:    %10 = stablehlo.select %c, %cst_1, %9 : tensor<3x3xi1>, tensor<3x3xf32>
// FORWARD-NEXT:    %11 = stablehlo.broadcast_in_dim %S, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %12 = stablehlo.multiply %1, %11 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<3x3xf32>
// FORWARD-NEXT:    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %14 = stablehlo.add %12, %13 : tensor<3x3xf32>
// FORWARD-NEXT:    %15 = stablehlo.multiply %10, %14 : tensor<3x3xf32>
// FORWARD-NEXT:    %16 = stablehlo.dot_general %U, %15, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %17 = stablehlo.broadcast_in_dim %S, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %18 = stablehlo.multiply %1, %17 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<3x3xf32>
// FORWARD-NEXT:    %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    %20 = stablehlo.add %18, %19 : tensor<3x3xf32>
// FORWARD-NEXT:    %21 = stablehlo.multiply %10, %20 : tensor<3x3xf32>
// FORWARD-NEXT:    %22 = stablehlo.dot_general %21, %Vt, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// FORWARD-NEXT:    return %U, %16, %S, %3, %Vt, %22 : tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<3x3xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @svd(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3x3xf32>) -> tensor<3x3xf32> {
// REVERSE-NEXT{LITERAL}:    %c = stablehlo.constant dense<[[true, false, false], [false, true, false], [false, false, true]]> : tensor<3x3xi1>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x3xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// REVERSE-NEXT:    %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<i32>)
// REVERSE-NEXT:    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %U, %0 : tensor<3x3xf32>
// REVERSE-NEXT:    %2 = stablehlo.dot_general %U, %arg1, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %4 = stablehlo.subtract %2, %3 : tensor<3x3xf32>
// REVERSE-NEXT:    %5 = stablehlo.broadcast_in_dim %S, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %6 = stablehlo.multiply %4, %5 : tensor<3x3xf32>
// REVERSE-NEXT:    %7 = stablehlo.dot_general %Vt, %arg3, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %9 = stablehlo.subtract %7, %8 : tensor<3x3xf32>
// REVERSE-NEXT:    %10 = stablehlo.broadcast_in_dim %S, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %11 = stablehlo.multiply %9, %10 : tensor<3x3xf32>
// REVERSE-NEXT:    %12 = stablehlo.add %6, %11 : tensor<3x3xf32>
// REVERSE-NEXT:    %13 = stablehlo.multiply %S, %S : tensor<3xf32>
// REVERSE-NEXT:    %14 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %15 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %16 = stablehlo.subtract %14, %15 : tensor<3x3xf32>
// REVERSE-NEXT:    %17 = stablehlo.select %c, %cst, %16 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %18 = stablehlo.divide %cst, %17 : tensor<3x3xf32>
// REVERSE-NEXT:    %19 = stablehlo.select %c, %cst_0, %18 : tensor<3x3xi1>, tensor<3x3xf32>
// REVERSE-NEXT:    %20 = stablehlo.multiply %19, %12 : tensor<3x3xf32>
// REVERSE-NEXT:    %21 = stablehlo.dot_general %U, %20, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    %22 = stablehlo.add %1, %21 : tensor<3x3xf32>
// REVERSE-NEXT:    %23 = stablehlo.dot_general %22, %Vt, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// REVERSE-NEXT:    return %23 : tensor<3x3xf32>
// REVERSE-NEXT:  }
