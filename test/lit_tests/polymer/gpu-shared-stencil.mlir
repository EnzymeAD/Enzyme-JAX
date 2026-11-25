// RUN: enzymexlamlir-opt %s --gpu-polyhedral-opt="block-size-x=16 block-size-y=1" | FileCheck %s

module {
  func.func @stencil_1d(%in: memref<66xf32>, %out: memref<66xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.for %i = 1 to 65 {
        %l = affine.load %in[%i - 1] : memref<66xf32>
        %c = affine.load %in[%i] : memref<66xf32>
        %r = affine.load %in[%i + 1] : memref<66xf32>
        %s0 = arith.addf %l, %c : f32
        %s1 = arith.addf %s0, %r : f32
        affine.store %s1, %out[%i] : memref<66xf32>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @stencil_1d
// CHECK: affine.parallel (%{{.+}}) = (0) to (5)
// CHECK: %[[SH:.+]] = memref.alloca() alignment = 16 : memref<18xf32>
// CHECK: affine.parallel (%{{.+}}) = (0) to (16)
// CHECK: affine.for %{{.+}} = {{.+}} to 18 step 16 {
// CHECK:   affine.load %arg0[{{.+}}] : memref<66xf32>
// CHECK:   affine.store %{{.+}}, %[[SH]][{{.+}}] : memref<18xf32>
// CHECK: }
// CHECK: "enzymexla.barrier"
// CHECK: affine.load %[[SH]][{{.+}}] : memref<18xf32>
// CHECK: affine.load %[[SH]][{{.+}}] : memref<18xf32>
// CHECK: affine.load %[[SH]][{{.+}}] : memref<18xf32>
// CHECK: affine.store %{{.+}}, %arg1[{{.+}}] : memref<66xf32>
