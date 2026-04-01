// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=compare_negate_const_simplify},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.negate %arg0 : tensor<f64>
    %1 = stablehlo.abs %0 : tensor<f64>
    %2 = stablehlo.negate %1 : tensor<f64>
    %3 = stablehlo.exponential %2 : tensor<f64>
    %4 = stablehlo.log_plus_one %3 : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %5 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %6 = stablehlo.compare  LT, %0, %5 : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.convert %cst : tensor<f64>
    %8 = stablehlo.select %6, %7, %0 : tensor<i1>, tensor<f64>
    %9 = stablehlo.add %4, %8 : tensor<f64>
    %10 = stablehlo.negate %9 : tensor<f64>
    return %10, %arg0 : tensor<f64>, tensor<f64>
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.negate %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.abs %0 : tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.negate %1 : tensor<f64>
// CHECK-NEXT:     %3 = stablehlo.exponential %2 : tensor<f64>
// CHECK-NEXT:     %4 = stablehlo.log_plus_one %3 : tensor<f64>
// CHECK-NEXT:     %5 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:     %6 = stablehlo.compare  LT, %0, %5 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT:     %7 = stablehlo.convert %cst : tensor<f64>
// CHECK-NEXT:     %8 = stablehlo.select %6, %7, %0 : tensor<i1>, tensor<f64>
// CHECK-NEXT:     %9 = stablehlo.add %4, %8 : tensor<f64>
// CHECK-NEXT:     %10 = stablehlo.negate %9 : tensor<f64>
// CHECK-NEXT:     return %10, %arg0 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT: }

