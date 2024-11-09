// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<5xf32>) -> tensor<10xf32> {
    %start = stablehlo.constant dense<0> : tensor<i64>
    %0 = "stablehlo.dynamic_update_slice"(%arg0, %arg1, %start) : (tensor<10xf32>, tensor<5xf32>, tensor<i64>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:    %0 = stablehlo.dynamic_update_slice %arg1, %arg3, %c : (tensor<10xf32>, tensor<5xf32>, tensor<i64>) -> tensor<10xf32>
// FORWARD-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %arg2, %c : (tensor<10xf32>, tensor<5xf32>, tensor<i64>) -> tensor<10xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<5xf32>, %arg2: tensor<10xf32>) -> (tensor<10xf32>, tensor<5xf32>) {
// REVERSE-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<5xf32>
// REVERSE-NEXT:    %0 = stablehlo.add %arg2, %cst : tensor<10xf32>
// REVERSE-NEXT:    %1 = stablehlo.dynamic_update_slice %0, %cst_0, %c : (tensor<10xf32>, tensor<5xf32>, tensor<i64>) -> tensor<10xf32>
// REVERSE-NEXT:    %2 = stablehlo.add %1, %cst : tensor<10xf32>
// REVERSE-NEXT:    %3 = stablehlo.dynamic_slice %0, %c, sizes = [5] : (tensor<10xf32>, tensor<i64>) -> tensor<5xf32>
// REVERSE-NEXT:    %4 = stablehlo.add %3, %cst_0 : tensor<5xf32>
// REVERSE-NEXT:    return %2, %4 : tensor<10xf32>, tensor<5xf32>
// REVERSE-NEXT:  }
