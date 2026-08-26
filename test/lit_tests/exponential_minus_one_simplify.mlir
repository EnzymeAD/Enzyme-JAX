// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=exponential_minus_one_fuse},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

// exp(x) - 1 -> expm1(x)
func.func @expm1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.0> : tensor<f64>
    %1 = stablehlo.subtract %0, %cst : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @expm1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.exponential_minus_one %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT: }

// 1 - exp(x) -> -expm1(x)
func.func @neg_expm1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.0> : tensor<f64>
    %1 = stablehlo.subtract %cst, %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @neg_expm1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.exponential_minus_one %arg0 : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.negate %0 : tensor<f64>
// CHECK-NEXT:    return %1 : tensor<f64>
// CHECK-NEXT: }

// exp(x) + -1 -> expm1(x)
func.func @expm1_add1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<-1.0> : tensor<f64>
    %1 = stablehlo.add %0, %cst : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @expm1_add1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.exponential_minus_one %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT: }

// -1 + exp(x) -> expm1(x)
func.func @expm1_add2(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<-1.0> : tensor<f64>
    %1 = stablehlo.add %cst, %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @expm1_add2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.exponential_minus_one %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT: }
