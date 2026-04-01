// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s

module {
  func.func @test(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
    %1 = stablehlo.multiply %arg1, %arg1 : tensor<f64>
    %2 = stablehlo.add %0, %1 : tensor<f64>
    %3 = stablehlo.add %2, %1 : tensor<f64>
    %4 = stablehlo.subtract %3, %2 : tensor<f64>
    return %4 : tensor<f64>
  }
}

// CHECK:  func.func @test(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg1 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg1, %arg1 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.add %0, %1 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %1 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
// CHECK-NEXT:    %4 = stablehlo.subtract %3, %2 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
// CHECK-NEXT:    return %4 : tensor<f64>
// CHECK-NEXT:  }
