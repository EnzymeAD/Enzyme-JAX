// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

module {
  func.func @main(%arg0: tensor<10x5xf32>) -> tensor<14x5xf32> {
    %0 = "enzymexla.extend"(%arg0) {
      lhs = 2 : i64,
      rhs = 2 : i64,
      dimension = 0 : i64
    } : (tensor<10x5xf32>) -> tensor<14x5xf32>
    return %0 : tensor<14x5xf32>
  }
}

// CHECK-LABEL: func.func @main(%arg0: tensor<10x5xf32>, %arg1: tensor<14x5xf32>) -> tensor<10x5xf32> {
// CHECK-DAG:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %cst_0 = arith.constant dense<0.000000e+00> : tensor<14x5xf32>
// CHECK-DAG:     %cst_1 = arith.constant dense<0.000000e+00> : tensor<10x5xf32>
// CHECK:         %0 = arith.addf %arg1, %cst_0 : tensor<14x5xf32>
// CHECK:         %1 = stablehlo.slice %0 [0:2, 0:5] : (tensor<14x5xf32>) -> tensor<2x5xf32>
// CHECK:         %2 = stablehlo.pad %1, %cst, low = [0, 0], high = [8, 0], interior = [0, 0] : (tensor<2x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK:         %3 = stablehlo.slice %0 [12:14, 0:5] : (tensor<14x5xf32>) -> tensor<2x5xf32>
// CHECK:         %4 = stablehlo.pad %3, %cst, low = [8, 0], high = [0, 0], interior = [0, 0] : (tensor<2x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK:         %5 = stablehlo.add %2, %4 : tensor<10x5xf32>
// CHECK:         %6 = stablehlo.slice %0 [2:12, 0:5] : (tensor<14x5xf32>) -> tensor<10x5xf32>
// CHECK:         %7 = stablehlo.add %5, %6 : tensor<10x5xf32>
// CHECK:         %8 = arith.addf %7, %cst_1 : tensor<10x5xf32>
// CHECK:         return %8 : tensor<10x5xf32>
