// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=noop_reverse" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main() -> tensor<8x4x3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<8x4x3xf32>
    %1 = stablehlo.reverse %cst, dims = [2, 1] : tensor<8x4x3xf32>
    return %1 : tensor<8x4x3xf32>
    // CHECK: %cst = stablehlo.constant dense<0.000000e+00> : tensor<8x4x3xf32>
    // CHECK-NEXT: return %cst : tensor<8x4x3xf32>
  }
}

module {
  func.func @main(%arg0: tensor<8x1xf32>) -> tensor<8x4x3x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 3] : (tensor<8x1xf32>) -> tensor<8x4x3x1xf32>
    %1 = stablehlo.reverse %0, dims = [3, 2, 0] : tensor<8x4x3x1xf32>
    return %1 : tensor<8x4x3x1xf32>

    // CHECK: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 3] : (tensor<8x1xf32>) -> tensor<8x4x3x1xf32>
    // CHECK-NEXT: %1 = stablehlo.reverse %0, dims = [0] : tensor<8x4x3x1xf32>
    // CHECK-NEXT: return %1 : tensor<8x4x3x1xf32>
  }
}

module {
  func.func @main(%arg0: tensor<8x1xf32>) -> tensor<8x4x3x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 3] : (tensor<8x1xf32>) -> tensor<8x4x3x1xf32>
    %1 = stablehlo.reverse %0, dims = [2, 0] : tensor<8x4x3x1xf32>
    return %1 : tensor<8x4x3x1xf32>

    // CHECK: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 3] : (tensor<8x1xf32>) -> tensor<8x4x3x1xf32>
    // CHECK-NEXT: %1 = stablehlo.reverse %0, dims = [0] : tensor<8x4x3x1xf32>
    // CHECK-NEXT: return %1 : tensor<8x4x3x1xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<1x8xf32>
    return %0 : tensor<1x8xf32>

    // CHECK: %0 = stablehlo.reverse %arg0, dims = [1] : tensor<1x8xf32>
    // CHECK-NEXT: return %0 : tensor<1x8xf32>
  }
}
