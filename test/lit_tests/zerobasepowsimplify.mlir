// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=no_nan_zero_base_pow_simplify(1)},transform-interpreter,enzyme-hlo-remove-transform,canonicalize,cse)" | FileCheck %s

func.func @main1(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %cst, %arg0 : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK: func.func @main1(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<10x4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:     %0 = stablehlo.compare  GT, %arg0, %cst_1 : (tensor<10x4xf32>, tensor<10x4xf32>) -> tensor<10x4xi1>
// CHECK-NEXT:     %1 = stablehlo.select %0, %cst_1, %cst_0 : tensor<10x4xi1>, tensor<10x4xf32>
// CHECK-NEXT:     %2 = stablehlo.compare  EQ, %arg0, %cst_1 : (tensor<10x4xf32>, tensor<10x4xf32>) -> tensor<10x4xi1>
// CHECK-NEXT:     %3 = stablehlo.select %2, %cst, %1 : tensor<10x4xi1>, tensor<10x4xf32>
// CHECK-NEXT:     return %3 : tensor<10x4xf32>
// CHECK-NEXT: }
