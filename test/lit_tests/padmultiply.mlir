// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: @pad_multiply
// CHECK:  %[[SLICE:.+]] = stablehlo.slice %{{.*}} [0:1, 0:3, 1024:2048]
// CHECK:  %[[MUL:.+]] = stablehlo.multiply %{{.*}}, %[[SLICE]] : tensor<1x3x1024xf32>
// CHECK:  stablehlo.pad %[[MUL]], %{{.*}}, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0]
func.func @pad_multiply(%4: tensor<1x3x1024xf32>, %2: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
  %constant_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %5 = stablehlo.pad %4, %constant_0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %7 = stablehlo.multiply %5, %2 : tensor<1x3x2048xf32>
  return %7 : tensor<1x3x2048xf32>
}
