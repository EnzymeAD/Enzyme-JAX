// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=log_simplify},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

// log(x + 1) -> log1p(x)
func.func @log1p_1(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<1.0> : tensor<f64>
    %0 = stablehlo.add %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @log1p_1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.log_plus_one %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT: }

// log(1 + x) -> log1p(x)
func.func @log1p_2(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<1.0> : tensor<f64>
    %0 = stablehlo.add %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @log1p_2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.log_plus_one %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT: }
