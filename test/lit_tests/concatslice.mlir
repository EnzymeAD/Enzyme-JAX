// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.slice %arg0 [0:4, 0:1] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %1 = stablehlo.slice %arg0 [0:4, 1:2] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %2 = stablehlo.slice %arg0 [0:4, 2:3] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %3 = stablehlo.slice %arg0 [0:4, 3:4] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %4 = stablehlo.concatenate %0, %1, %2, %3, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    return %arg0 : tensor<4x4xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.slice %arg0 [0:4, 0:1] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %1 = stablehlo.slice %arg0 [0:4, 1:2] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %2 = stablehlo.slice %arg0 [0:4, 2:3] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %3 = stablehlo.slice %arg0 [0:4, 2:3] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %4 = stablehlo.concatenate %0, %1, %2, %3, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:4, 2:3] : (tensor<4x4xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:4, 0:3] : (tensor<4x4xf32>) -> tensor<4x3xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, %0, dim = 1 : (tensor<4x3xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %2 : tensor<4x4xf32>
// CHECK-NEXT:  }
