// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_to_trmm;fuse_mul_into_trmm" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main1(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x32xf64> {enzymexla.memory_effects = []}) -> tensor<64x32xf64> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
    %3 = stablehlo.compare  NE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %5 = stablehlo.select %4, %0, %cst_0 : tensor<64x64xi1>, tensor<64x64xf64>
    %6 = stablehlo.select %3, %5, %cst : tensor<64x64xi1>, tensor<64x64xf64>
    %7 = stablehlo.dot_general %6, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x64xf64>, tensor<64x32xf64>) -> tensor<64x32xf64>
    return %7 : tensor<64x32xf64>
}

// CHECK:  func.func @main1(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x32xf64> {enzymexla.memory_effects = []}) -> tensor<64x32xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
// CHECK-NEXT:    %3 = stablehlo.compare  NE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %5 = stablehlo.select %4, %0, %cst_1 {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    %6 = stablehlo.select %3, %5, %cst_0 {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>], enzymexla.upper_unit_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    %7 = enzymexla.blas.trmm %6, %arg1, %cst {side = #enzymexla.side<left>, transpose = #enzymexla.transpose<none>, unit_diagonal, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf64>, tensor<64x32xf64>, tensor<f64>) -> tensor<64x32xf64>
// CHECK-NEXT:    return %7 : tensor<64x32xf64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x32xf64> {enzymexla.memory_effects = []}) -> tensor<32x64xf64> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
    %cst_c = stablehlo.constant dense<3.000000e+00> : tensor<32x64xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %5 = stablehlo.select %4, %0, %cst_0 : tensor<64x64xi1>, tensor<64x64xf64>
    %6 = stablehlo.dot_general %arg1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf64>, tensor<64x64xf64>) -> tensor<32x64xf64>
    %7 = stablehlo.multiply %6, %cst_c : tensor<32x64xf64>
    return %7 : tensor<32x64xf64>
}

// CHECK:  func.func @main2(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x32xf64> {enzymexla.memory_effects = []}) -> tensor<32x64xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
// CHECK-NEXT:    %3 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %4 = stablehlo.select %3, %0, %cst_1 {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>], enzymexla.upper_unit_tri_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    %5 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<64x32xf64>) -> tensor<32x64xf64>
// CHECK-NEXT:    %6 = stablehlo.multiply %cst_0, %cst : tensor<f64>
// CHECK-NEXT:    %7 = enzymexla.blas.trmm %4, %5, %6 {side = #enzymexla.side<right>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf64>, tensor<32x64xf64>, tensor<f64>) -> tensor<32x64xf64>
// CHECK-NEXT:    return %7 : tensor<32x64xf64>
// CHECK-NEXT:  }