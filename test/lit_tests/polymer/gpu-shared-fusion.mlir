// RUN: enzymexlamlir-opt %s --gpu-polyhedral-opt="block-size-x=16 block-size-y=1 tile-size=16" | FileCheck %s

module {
  func.func @fused_producer_consumer(%X: memref<4x16xf32>, %T: memref<4x16xf32>, %W: memref<16x16xf32>, %Y: memref<4x16xf32>) {
    %c1 = arith.constant 1 : index
    %cst2 = arith.constant 2.0 : f32
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.for %b = 0 to 4 {
        affine.for %j = 0 to 16 {
          %x = affine.load %X[%b, %j] : memref<4x16xf32>
          %scaled = arith.mulf %x, %cst2 : f32
          affine.store %scaled, %T[%b, %j] : memref<4x16xf32>
        }
      }
      affine.for %b = 0 to 4 {
        affine.for %i = 0 to 16 {
          affine.for %k = 0 to 16 {
            %y = affine.load %Y[%b, %i] : memref<4x16xf32>
            %t = affine.load %T[%b, %k] : memref<4x16xf32>
            %w = affine.load %W[%i, %k] : memref<16x16xf32>
            %p = arith.mulf %t, %w : f32
            %ny = arith.addf %y, %p : f32
            affine.store %ny, %Y[%b, %i] : memref<4x16xf32>
          }
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @fused_producer_consumer
// CHECK: affine.parallel (%{{.+}}) = (0) to (4)
// CHECK-DAG: memref.alloca() alignment = 16 : memref<1x16xf32>
// CHECK: affine.parallel (%{{.+}}) = (0) to (16)
// Intermediate tile %arg1 (%T) is written in shared memory before being read,
// so no global load from %arg1 is emitted!
// CHECK-NOT: affine.load %arg1
// CHECK: "enzymexla.barrier"
// CHECK-NOT: affine.load %arg1
// CHECK: return
