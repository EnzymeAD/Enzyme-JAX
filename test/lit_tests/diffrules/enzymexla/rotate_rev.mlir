// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rotate outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rotate_large_amount outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=CHECK-LARGE

module {
  func.func @rotate(%arg0: tensor<10x5xf32>) -> tensor<10x5xf32> {
    %0 = "enzymexla.rotate"(%arg0) {
      dimension = 1 : i32,
      amount = 2 : i32
    } : (tensor<10x5xf32>) -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
  }
  func.func @rotate_large_amount(%arg0: tensor<10x5xf32>) -> tensor<10x5xf32> {
    %0 = "enzymexla.rotate"(%arg0) {
      dimension = 1 : i32,
      amount = 7 : i32
    } : (tensor<10x5xf32>) -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
  }
}

// CHECK-LABEL: func.func @rotate(%arg0: tensor<10x5xf32>, %arg1: tensor<10x5xf32>) -> tensor<10x5xf32> {
// CHECK:         %0 = "enzymexla.rotate"(%arg1) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<10x5xf32>) -> tensor<10x5xf32>
// CHECK:         return %0 : tensor<10x5xf32>

// CHECK-LARGE-LABEL: func.func @rotate_large_amount(%arg0: tensor<10x5xf32>, %arg1: tensor<10x5xf32>) -> tensor<10x5xf32> {
// CHECK-LARGE:         %0 = "enzymexla.rotate"(%arg1) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<10x5xf32>) -> tensor<10x5xf32>
// CHECK-LARGE:         return %0 : tensor<10x5xf32>
