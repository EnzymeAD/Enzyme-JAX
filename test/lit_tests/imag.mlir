// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)  {
    %0 = "stablehlo.complex"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xcomplex<f32>>
    %1 = "stablehlo.imag"(%0) : (tensor<10xcomplex<f32>>) -> tensor<10xf32>
    %2 = "stablehlo.imag"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
    return %1, %2 : tensor<10xf32>, tensor<10xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// CHECK-NEXT:    return %arg0, %cst : tensor<10xf32>, tensor<10xf32>
// CHECK-NEXT:  }
