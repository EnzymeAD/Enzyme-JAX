
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

// A reshape that only inserts unit dimensions of a concatenate other ops
// read stays: it is a view of the concatenate, a concatenate of reshaped
// operands would be a second copy.
func.func @shared_concat(%a: tensor<2x2x2x1xi64>, %b: tensor<2x2x2x1xi64>) -> (tensor<2x2x2x2xi64>, tensor<1x2x2x2x2xi64>) {
  %0 = stablehlo.concatenate %a, %b, dim = 3 : (tensor<2x2x2x1xi64>, tensor<2x2x2x1xi64>) -> tensor<2x2x2x2xi64>
  %1 = stablehlo.reshape %0 : (tensor<2x2x2x2xi64>) -> tensor<1x2x2x2x2xi64>
  return %0, %1 : tensor<2x2x2x2xi64>, tensor<1x2x2x2x2xi64>
}

// CHECK-LABEL: func.func @shared_concat(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<2x2x2x1xi64>, %[[B:[^ ,:)]+]]: tensor<2x2x2x1xi64>) -> (tensor<2x2x2x2xi64>, tensor<1x2x2x2x2xi64>) {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.concatenate %[[A]], %[[B]], dim = 3 : (tensor<2x2x2x1xi64>, tensor<2x2x2x1xi64>) -> tensor<2x2x2x2xi64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reshape %[[C]] : (tensor<2x2x2x2xi64>) -> tensor<1x2x2x2x2xi64>
// CHECK-NEXT: return %[[C]], %[[R]] : tensor<2x2x2x2xi64>, tensor<1x2x2x2x2xi64>
// CHECK-NEXT: }
