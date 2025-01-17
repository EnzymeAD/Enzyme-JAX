// RUN: enzymexlamlir-opt --enzyme-hlo-opt --cse %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<1x3x2x2xf32>) -> tensor<2x2x3x1xf32>
    %1 = stablehlo.cosine %0 : tensor<2x2x3x1xf32>
    %2 = stablehlo.transpose %1, dims = [3, 2, 1, 0] : (tensor<2x2x3x1xf32>) -> tensor<1x3x2x2xf32>
    return %2 : tensor<1x3x2x2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.cosine %arg0 : tensor<1x3x2x2xf32>
// CHECK-NEXT:    return %0 : tensor<1x3x2x2xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<1x3x2x2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2x2x3x1xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<1x3x2x2xf32>) -> tensor<2x2x3x1xf32>
    %1 = stablehlo.cosine %0 : tensor<2x2x3x1xf32>
    %2 = stablehlo.transpose %1, dims = [3, 2, 1, 0] : (tensor<2x2x3x1xf32>) -> tensor<1x3x2x2xf32>
    %3 = stablehlo.sine %1 : tensor<2x2x3x1xf32>
    return %2, %3 : tensor<1x3x2x2xf32>, tensor<2x2x3x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x3x2x2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2x2x3x1xf32>) {
// CHECK-NEXT:    %0 = stablehlo.cosine %arg0 : tensor<1x3x2x2xf32>
// CHECK-NEXT:    %1 = stablehlo.sine %0 : tensor<1x3x2x2xf32>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [3, 2, 1, 0] : (tensor<1x3x2x2xf32>) -> tensor<2x2x3x1xf32>
// CHECK-NEXT:    return %0, %2 : tensor<1x3x2x2xf32>, tensor<2x2x3x1xf32>
// CHECK-NEXT:  }
