
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_concat" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  // CHECK-LABEL: @test_reshape_of_concat_1
  func.func @test_reshape_of_concat_1(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<4xf32> {
    // CHECK: reshape
    // CHECK-NEXT: reshape
    // CHECK-NEXT: concatenate
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x4xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
