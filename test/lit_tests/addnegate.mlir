// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.negate %arg0 : tensor<4x4xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.subtract %arg1, %arg0 : tensor<4x4xf32>
// CHECK-NEXT:    return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }
