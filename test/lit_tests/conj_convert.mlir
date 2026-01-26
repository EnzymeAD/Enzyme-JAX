// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @conj_1(%arg0: tensor<64x64xf64>) -> tensor<64x64xcomplex<f64>> {
    %120 = stablehlo.convert %arg0 : (tensor<64x64xf64>) -> tensor<64x64xcomplex<f64>>
    %121 = chlo.conj %120 : tensor<64x64xcomplex<f64>> -> tensor<64x64xcomplex<f64>>
    return %121 : tensor<64x64xcomplex<f64>>
}

// CHECK: func.func @conj_1(%arg0: tensor<64x64xf64>) -> tensor<64x64xcomplex<f64>> {
// CHECK-NEXT:     %0 = stablehlo.convert %arg0 : (tensor<64x64xf64>) -> tensor<64x64xcomplex<f64>>
// CHECK-NEXT:     return %0 : tensor<64x64xcomplex<f64>>
// CHECK-NEXT: }

func.func @conj_2(%arg0: tensor<64x64xf64>) -> tensor<64x64xf64> {
    %120 = stablehlo.convert %arg0 : (tensor<64x64xf64>) -> tensor<64x64xcomplex<f64>>
    %121 = chlo.conj %120 : tensor<64x64xcomplex<f64>> -> tensor<64x64xcomplex<f64>>
    %122 = stablehlo.multiply %120, %121 : tensor<64x64xcomplex<f64>>
    %123 = stablehlo.real %122 : (tensor<64x64xcomplex<f64>>) -> tensor<64x64xf64>
    return %123 : tensor<64x64xf64>
}

// CHECK: func.func @conj_2(%arg0: tensor<64x64xf64>) -> tensor<64x64xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg0 : tensor<64x64xf64>
// CHECK-NEXT:     return %0 : tensor<64x64xf64>
// CHECK-NEXT: }
