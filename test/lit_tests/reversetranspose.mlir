// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reverse_transpose" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
    // CHECK-DAG: %[[REVERSE:.+]] = stablehlo.reverse %arg0, dims = [0, 2] : tensor<2x3x4xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
    // CHECK-DAG: %[[TRANSPOSE:.+]] = stablehlo.transpose %[[REVERSE]], dims = [1, 0, 2] : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
    %1 = stablehlo.reverse %0, dims = [1, 2] : (tensor<3x2x4xf32>) -> tensor<3x2x4xf32>
    return %1 : tensor<3x2x4xf32>
  }
}
