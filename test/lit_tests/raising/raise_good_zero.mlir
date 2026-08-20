// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt | FileCheck %s

module {
  func.func @kernel_with_poison(%arg0: memref<f32>) {

    %0 = ub.poison : f32
    affine.store %0, %arg0[] : memref<f32>
    return

  }
}

// CHECK:  func.func private @kernel_with_poison_raised(%arg0: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    return %[[ZERO]] : tensor<f32>
// CHECK-NEXT:  }
