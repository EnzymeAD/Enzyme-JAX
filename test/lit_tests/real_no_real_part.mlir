// RUN: enzymexlamlir-opt %s -enzyme-hlo-opt | FileCheck %s

module {
  func.func @test(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[(0.00e+00, 1.00e+00), (0.00e+00, 2.00e+00)]> : tensor<2xcomplex<f32>>
    %cst2 = stablehlo.constant dense<0.00e+00> : tensor<2xf32>
    %0 = stablehlo.complex %cst2, %arg0 : tensor<2xcomplex<f32>>
    %1 = stablehlo.add %0, %cst : tensor<2xcomplex<f32>>
    %2 = stablehlo.real %1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    return %2 : tensor<2xf32>
  }
}

// CHECK: func.func @test(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:     return %cst : tensor<2xf32>
// CHECK-NEXT: }

module {
  func.func @test_fail(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[(2.00e+00, 1.00e+00), (0.00e+00, 2.00e+00)]> : tensor<2xcomplex<f32>>
    %cst2 = stablehlo.constant dense<0.00e+00> : tensor<2xf32>
    %0 = stablehlo.complex %cst2, %arg0 : tensor<2xcomplex<f32>>
    %1 = stablehlo.add %0, %cst : tensor<2xcomplex<f32>>
    %2 = stablehlo.real %1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    return %2 : tensor<2xf32>
  }
}

// CHECK: func.func @test_fail(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>]} dense<[(2.000000e+00,1.000000e+00), (0.000000e+00,2.000000e+00)]> : tensor<2xcomplex<f32>>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:   %0 = stablehlo.complex %cst_0, %arg0 {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed GUARANTEED>], enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// CHECK-NEXT:   %1 = stablehlo.add %0, %cst {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// CHECK-NEXT:   %2 = stablehlo.real %1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT:   return %2 : tensor<2xf32>
// CHECK-NEXT: }
