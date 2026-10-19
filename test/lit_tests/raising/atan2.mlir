// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise --raise-affine-to-stablehlo --enzyme-hlo-opt --arith-raise | FileCheck %s

module {
  llvm.func @__nv_atan2f(f32, f32) -> f32

  func.func @atan2_elementwise(%y: memref<64xf32>, %x: memref<64xf32>, %out: memref<64xf32>) {
    affine.parallel (%i) = (0) to (64) {
      %yv = affine.load %y[%i] : memref<64xf32>
      %xv = affine.load %x[%i] : memref<64xf32>
      %r = llvm.call @__nv_atan2f(%yv, %xv) : (f32, f32) -> f32
      affine.store %r, %out[%i] : memref<64xf32>
    }
    return
  }
}

// CHECK:  func.func private @atan2_elementwise_raised(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
// CHECK-NEXT:    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<64xf32>
// CHECK-NEXT:    return %arg0, %arg1, %0 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
// CHECK-NEXT:  }
