// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_reverse" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
    // CHECK-DAG: %[[TRANSPOSE:.+]] = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
    %1 = stablehlo.reverse %arg0, dims = [1, 2] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK-DAG: %[[REVERSE:.+]] = stablehlo.reverse %[[TRANSPOSE]], dims = [0, 2] : tensor<3x2x4xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
    return %2 : tensor<3x2x4xf32>
  }
}
