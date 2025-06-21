// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=common_associative_commutative_op_reorder},transform-interpreter,enzyme-hlo-remove-transform,canonicalize,cse,enzyme-hlo-generate-td{patterns=common_associative_commutative_op_reorder},transform-interpreter,enzyme-hlo-remove-transform,cse,canonicalize)" | FileCheck %s

func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
    return %3 : tensor<f64>
}

// CHECK: func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
// CHECK-NEXT:    return %2 : tensor<f64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
    %4 = stablehlo.multiply %arg0, %0 : tensor<f64>
    %5 = stablehlo.multiply %4, %3 : tensor<f64>
    return %5 : tensor<f64>
}

// CHECK: func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %1 : tensor<f64>
// CHECK-NEXT:    return %2 : tensor<f64>
// CHECK-NEXT:  }
