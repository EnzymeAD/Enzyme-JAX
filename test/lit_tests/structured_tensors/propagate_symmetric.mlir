  // RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %alpha = stablehlo.constant dense<2.0> : tensor<f32>
  %beta = stablehlo.constant dense<3.0> : tensor<f32>
  %0 = enzymexla.lapack.symm %arg0, %arg1, %arg2, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>, tensor<f32>) -> tensor<2x2xf32>
  %1 = stablehlo.reshape %arg0 {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = stablehlo.subtract %1, %0 : tensor<2x2xf32>
  %3 = stablehlo.dot_general %2, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %4 : tensor<2x2xf32>
}

// CHECK: func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = enzymexla.lapack.symm %arg0, %arg1, %arg2, %cst, %cst_0 {enzymexla.guaranteed_symmetric = true, side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %0 : tensor<2x2xf32>
// CHECK-NEXT:   %3 = stablehlo.dot_general %2, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %3 : tensor<2x2xf32>
// CHECK-NEXT: }
