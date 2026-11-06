// RUN: enzymexlamlir-opt %s --gpu-polyhedral-opt="block-size-x=16 block-size-y=16" | FileCheck %s

module {
  func.func @transpose(%in: memref<64x64xf32>, %out: memref<64x64xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.for %i = 0 to 64 {
        affine.for %j = 0 to 64 {
          %v = affine.load %in[%i, %j] : memref<64x64xf32>
          affine.store %v, %out[%j, %i] : memref<64x64xf32>
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @transpose
// CHECK: affine.parallel (%{{.+}}, %{{.+}}) = (0, 0) to (4, 4)
// CHECK: %[[SH:.+]] = memref.alloca() alignment = 16 : memref<16x16xf32>
// CHECK: affine.parallel (%{{.+}}, %{{.+}}) = (0, 0) to (16, 16)
// CHECK: "enzymexla.barrier"
