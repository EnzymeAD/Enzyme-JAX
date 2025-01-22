// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup,enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%0: tensor<32x1xf32>, %1: tensor<i64>, %2: tensor<i64>) -> tensor<32x1xf32> {
    %res = stablehlo.dynamic_slice %0, %1, %2, sizes = [32, 1] : (tensor<32x1xf32>, tensor<i64>, tensor<i64>) -> tensor<32x1xf32>
    return %res : tensor<32x1xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<32x1xf32>, %arg1: tensor<32x1xf32>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>) -> (tensor<32x1xf32>, tensor<32x1xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.dynamic_slice %arg1, %arg2, %arg4, sizes = [32, 1] : (tensor<32x1xf32>, tensor<i64>, tensor<i64>) -> tensor<32x1xf32>
// FORWARD-NEXT:    %1 = stablehlo.dynamic_slice %arg0, %arg2, %arg4, sizes = [32, 1] : (tensor<32x1xf32>, tensor<i64>, tensor<i64>) -> tensor<32x1xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<32x1xf32>, tensor<32x1xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<32x1xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<32x1xf32>) -> tensor<32x1xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x1xf32>
// REVERSE-NEXT:    %0 = stablehlo.dynamic_update_slice %cst, %arg3, %arg1, %arg2 : (tensor<32x1xf32>, tensor<32x1xf32>, tensor<i64>, tensor<i64>) -> tensor<32x1xf32>
// REVERSE-NEXT:    return %0 : tensor<32x1xf32>
// REVERSE-NEXT:  }