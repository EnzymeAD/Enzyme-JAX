// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dot_general_to_syrk;transpose_syrk_to_syrk;fuse_mul_into_syrk;fuse_add_into_syrk},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

func.func @main1(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
  %cst = stablehlo.constant dense<5.000000e-01> : tensor<64x64xf32>
  %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
  %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
  %1 = stablehlo.multiply %0, %cst : tensor<64x64xf32>
  %2 = stablehlo.add %1, %cst_1 : tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}

// CHECK: func.func @main1(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<5.000000e-01> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:   %0 = enzymexla.blas.syrk %arg0, %cst_1, %cst_0, %cst {fill, uplo = #enzymexla.uplo<F>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %0 : tensor<64x64xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
  %cst = stablehlo.constant dense<5.000000e-01> : tensor<64x64xf32>
  %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x32xf32>) -> tensor<32x64xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0] : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<64x64xf32>
  %2 = stablehlo.multiply %1, %cst : tensor<64x64xf32>
  %3 = stablehlo.add %2, %cst_1 : tensor<64x64xf32>
  return %3 : tensor<64x64xf32>
}

// CHECK: func.func @main2(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<5.000000e-01> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:   %0 = enzymexla.blas.syrk %arg0, %cst_1, %cst_0, %cst {fill, uplo = #enzymexla.uplo<F>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %0 : tensor<64x64xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
  %cst = stablehlo.constant dense<5.000000e-01> : tensor<64x64xf32>
  %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x32xf32>) -> tensor<32x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x64xf32>) -> tensor<64x64xf32>
  %2 = stablehlo.multiply %1, %cst : tensor<64x64xf32>
  %3 = stablehlo.add %2, %cst_1 : tensor<64x64xf32>
  return %3 : tensor<64x64xf32>
}

// CHECK: func.func @main3(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<5.000000e-01> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:   %0 = enzymexla.blas.syrk %arg0, %cst_1, %cst_0, %cst {fill, uplo = #enzymexla.uplo<F>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %0 : tensor<64x64xf32>
// CHECK-NEXT: }


func.func @main4(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
  %cst = stablehlo.constant dense<5.000000e-01> : tensor<64x64xf32>
  %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x32xf32>) -> tensor<32x64xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [0] x [1] : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
  %2 = stablehlo.multiply %1, %cst : tensor<64x64xf32>
  %3 = stablehlo.add %2, %cst_1 : tensor<64x64xf32>
  return %3 : tensor<64x64xf32>
}

// CHECK: func.func @main4(%arg0: tensor<64x32xf32>) -> tensor<64x64xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed GUARANTEED>]} dense<5.000000e-01> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:   %0 = enzymexla.blas.syrk %arg0, %cst_1, %cst_0, %cst {fill, uplo = #enzymexla.uplo<F>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %0 : tensor<64x64xf32>
// CHECK-NEXT: }

func.func @fail1(%arg0: tensor<5x2xf32>) -> tensor<f32> {
  %0 = stablehlo.reshape %arg0 : (tensor<5x2xf32>) -> tensor<10xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10xf32>, tensor<10xf32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK: func.func @fail1(%arg0: tensor<5x2xf32>) -> tensor<f32> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<5x2xf32>) -> tensor<10xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10xf32>, tensor<10xf32>) -> tensor<f32>
// CHECK-NEXT:   return %1 : tensor<f32>
// CHECK-NEXT: }
