// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.compare  GE, %arg0, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = stablehlo.select %0, %arg1, %arg3 : tensor<i1>, tensor<f32>
    %2 = stablehlo.select %0, %arg0, %arg2 : tensor<i1>, tensor<f32>
    %3 = stablehlo.compare  LT, %arg0, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.select %3, %arg3, %arg1 : tensor<i1>, tensor<f32>
    %5 = stablehlo.maximum %arg0, %arg2 : tensor<f32>
    return %4, %5 : tensor<f32>, tensor<f32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
// CHECK-NEXT:    %0 = stablehlo.compare  GE, %arg0, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:    %1 = stablehlo.select %0, %arg1, %arg3 : tensor<i1>, tensor<f32>
// CHECK-NEXT:    %2 = stablehlo.maximum %arg0, %arg2 : tensor<f32>
// CHECK-NEXT:    return %1, %2 : tensor<f32>, tensor<f32>
// CHECK-NEXT:  }
