// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = stablehlo.not %0 : tensor<i1>
    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<i1>, tensor<f32>
    return %2 : tensor<f32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %0 = stablehlo.minimum %arg1, %arg0 : tensor<f32>
// CHECK-NEXT:    return %0 : tensor<f32>
// CHECK-NEXT:  }
