// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @transpose(%1845: tensor<32x32768x314xf32>) -> tensor<100x14x2xf32> {
  %1846 = stablehlo.transpose %1845, dims = [1, 2, 0] : (tensor<32x32768x314xf32>) -> tensor<32768x314x32xf32>
  %1847 = stablehlo.slice %1846 [4:3004:30, 300:314:1, 2:10:4] : (tensor<32768x314x32xf32>) -> tensor<100x14x2xf32>
  return %1847 : tensor<100x14x2xf32>
}

// CHECK:  func.func @transpose(%arg0: tensor<32x32768x314xf32>) -> tensor<100x14x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [2:10:4, 4:3004:30, 300:314] : (tensor<32x32768x314xf32>) -> tensor<2x100x14xf32>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<2x100x14xf32>) -> tensor<100x14x2xf32>
// CHECK-NEXT:    return %1 : tensor<100x14x2xf32>
// CHECK-NEXT:  }
