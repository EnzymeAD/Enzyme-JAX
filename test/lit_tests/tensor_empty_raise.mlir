// RUN: enzymexlamlir-opt %s --tensor-empty-raise | FileCheck %s

module {
  func.func @main(%s: index) -> (tensor<10xf32>, tensor<0xf32>, tensor<?xf32>) {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = tensor.empty() : tensor<0xf32>
    %2 = tensor.empty(%s) : tensor<?xf32> // not raised
    return %0, %1, %2 : tensor<10xf32>, tensor<0xf32>, tensor<?xf32>
  }
}

// CHECK:  func.func @main(%arg0: index) -> (tensor<10xf32>, tensor<0xf32>, tensor<?xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<0xf32>
// CHECK-NEXT:    %0 = tensor.empty(%arg0) : tensor<?xf32>
// CHECK-NEXT:    return %cst, %cst_0, %0 : tensor<10xf32>, tensor<0xf32>, tensor<?xf32>
// CHECK-NEXT:  }
