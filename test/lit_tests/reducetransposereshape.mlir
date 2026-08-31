// RUN: enzymexlamlir-opt %s "--enzyme-hlo-generate-td=patterns=full_reduce_reshape_or_transpose" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<20xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>

    %ts = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
    %rs = stablehlo.reshape %ts : (tensor<5x4xf32>) -> tensor<20xf32>
    %add = stablehlo.add %rs, %arg1 : tensor<20xf32>

    %v =  stablehlo.reshape %add : (tensor<20xf32>) -> tensor<4x5xf32>
    %red = stablehlo.reduce(%v init: %cst) applies stablehlo.add across dimensions = [0, 1]
        : (tensor<4x5xf32>, tensor<f32>) -> tensor<f32>
    return %red : tensor<f32>
  }

}

// CHECK:  func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<20xf32>) -> tensor<f32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<5x4xf32>) -> tensor<20xf32>
// CHECK-NEXT:    %2 = stablehlo.add %1, %arg1 : tensor<20xf32>
// CHECK-NEXT:    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<20xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %3 : tensor<f32>
// CHECK-NEXT:  }
