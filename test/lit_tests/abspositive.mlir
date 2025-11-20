// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @test1() -> tensor<18500xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<18500xi64>
    %1 = stablehlo.constant dense<1> : tensor<18500xi64>
    %2 = stablehlo.add %0, %1 : tensor<18500xi64>
    %3 = stablehlo.abs %2 : tensor<18500xi64>
    return %3 : tensor<18500xi64>
}

// CHECK:  func.func @test1() -> tensor<18500xi64> {
// CHECK-NEXT:    %c = stablehlo.constant {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<1> : tensor<18500xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<18500xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<18500xi64>
// CHECK-NEXT:    return %1 : tensor<18500xi64>
// CHECK-NEXT:  }

func.func @test2(%arg0: tensor<12xf64>) -> tensor<12xf64> {
    %0 = stablehlo.abs %arg0 : tensor<12xf64>
    %1 = stablehlo.abs %0 : tensor<12xf64>
    return %1 : tensor<12xf64>
}

// CHECK:  func.func @test2(%arg0: tensor<12xf64>) -> tensor<12xf64> {
// CHECK-NEXT:    %0 = stablehlo.abs %arg0 {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<12xf64>
// CHECK-NEXT:    return %0 : tensor<12xf64>
// CHECK-NEXT:  }

func.func @test3(%arg0: tensor<12xf64>) -> tensor<12xf64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<12xf64>
    %1 = stablehlo.abs %0 : tensor<12xf64>
    return %1 : tensor<12xf64>
}

// CHECK:  func.func @test3(%arg0: tensor<12xf64>) -> tensor<12xf64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<12xf64>
// CHECK-NEXT:    return %0 : tensor<12xf64>
// CHECK-NEXT:  }

func.func @test4(%arg0: tensor<12xf64>) -> tensor<4x3xf64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<12xf64>
    %1 = stablehlo.reshape %0 : (tensor<12xf64>) -> tensor<4x3xf64>
    %2 = stablehlo.abs %1 : tensor<4x3xf64>
    return %2 : tensor<4x3xf64>
}

// CHECK:  func.func @test4(%arg0: tensor<12xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<12xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<12xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    return %1 : tensor<4x3xf64>
// CHECK-NEXT:  }

func.func @test5(%arg0: tensor<12xf64>, %arg1: tensor<12xf64>, %arg2: tensor<12xf64>) -> tensor<12xf64> {
    // CHECK: chlo.erf_inv %arg0 {enzymexla.guaranteed_non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]}
    %0 = chlo.erf_inv %arg0 : tensor<12xf64> -> tensor<12xf64>
    %1 = stablehlo.add %arg1, %arg2 : tensor<12xf64>
    %2 = stablehlo.add %0, %1 : tensor<12xf64>
    // CHECK: stablehlo.abs
    %3 = stablehlo.abs %2 : tensor<12xf64>
    return %3 : tensor<12xf64>
}

func.func @test6(%arg0: tensor<12xf64>, %arg1: tensor<12xf64>, %arg2: tensor<12xf64>) -> tensor<12xf64> {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<12xf64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<12xf64>
    %1 = stablehlo.add %arg1, %arg2 : tensor<12xf64>
    %2 = stablehlo.add %0, %1 : tensor<12xf64>
    // CHECK: stablehlo.abs
    %3 = stablehlo.abs %2 : tensor<12xf64>
    return %3 : tensor<12xf64>
}
