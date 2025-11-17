  // RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %alpha = stablehlo.constant dense<2.0> : tensor<f32>
  %beta = stablehlo.constant dense<3.0> : tensor<f32>
  %c = stablehlo.constant dense<[[4.0, 3.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %0 = enzymexla.lapack.symm %c, %arg0, %arg1, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>, tensor<f32>) -> tensor<2x2xf32>
  %1 = stablehlo.subtract %0, %c : tensor<2x2xf32>
  %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}

// CHECK: func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant {enzymexla.guaranteed_symmetric = true} dense<{{\[\[}}4.000000e+00, 3.000000e+00], [3.000000e+00, 4.000000e+00{{\]\]}}> : tensor<2x2xf32>
// CHECK-NEXT:   %0 = enzymexla.lapack.symm %cst_1, %arg0, %arg1, %cst, %cst_0 {enzymexla.guaranteed_symmetric = true, side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %0, %cst_1 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }
