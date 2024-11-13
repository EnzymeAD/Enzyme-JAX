// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)  {
    %0 = "stablehlo.complex"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xcomplex<f32>>
    %1 = "stablehlo.real"(%0) : (tensor<10xcomplex<f32>>) -> tensor<10xf32>
    %2 = "stablehlo.real"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
    return %1, %2 : tensor<10xf32>, tensor<10xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// CHECK-NEXT:    return %arg0, %arg0 : tensor<10xf32>, tensor<10xf32>
// CHECK-NEXT:  }
