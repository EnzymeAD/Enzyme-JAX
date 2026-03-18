// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rotate outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

module {
  func.func @rotate(%arg0: tensor<10x5xf32>) -> tensor<10x5xf32> {
    %0 = "enzymexla.rotate"(%arg0) {
      dimension = 1 : i32,
      amount = 2 : i32
    } : (tensor<10x5xf32>) -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
  }
}

// CHECK-LABEL: func.func @rotate(%arg0: tensor<10x5xf32>, %arg1: tensor<10x5xf32>) -> tensor<10x5xf32> {
// CHECK-DAG:     %cst = arith.constant dense<0.000000e+00> : tensor<10x5xf32>
// CHECK:         %0 = arith.addf %arg1, %cst : tensor<10x5xf32>
// CHECK:         %1 = "enzymexla.rotate"(%0) <{amount = -2 : i32, dimension = 1 : i32}> : (tensor<10x5xf32>) -> tensor<10x5xf32>
// CHECK:         %2 = arith.addf %1, %cst : tensor<10x5xf32>
// CHECK:         return %2 : tensor<10x5xf32>
