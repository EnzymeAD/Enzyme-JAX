// RUN: enzymexlamlir-opt %s --gpu-polyhedral-opt="block-size-x=16 block-size-y=16 tile-size=16" | FileCheck %s

module {
  func.func @matmul(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.for %i = 0 to 64 {
        affine.for %j = 0 to 64 {
          affine.for %k = 0 to 64 {
            %a = affine.load %A[%i, %k] : memref<64x64xf32>
            %b = affine.load %B[%k, %j] : memref<64x64xf32>
            %c = affine.load %C[%i, %j] : memref<64x64xf32>
            %prod = arith.mulf %a, %b : f32
            %sum = arith.addf %c, %prod : f32
            affine.store %sum, %C[%i, %j] : memref<64x64xf32>
          }
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @matmul
// CHECK: "enzymexla.gpu_wrapper"
// CHECK: affine.parallel (%{{.+}}, %{{.+}}) = (0, 0) to (4, 4)
// CHECK-DAG: memref.alloca() alignment = 16 : memref<16x16xf32>
// CHECK-DAG: memref.alloca() alignment = 16 : memref<16x16xf32>
// CHECK: affine.parallel (%{{.+}}, %{{.+}}) = (0, 0) to (16, 16)
// CHECK: memref.alloca() : memref<f32>
// CHECK: affine.load %{{.+}}[{{.+}}] : memref<64x64xf32>
// CHECK: affine.store %{{.+}}, %{{.+}}[] : memref<f32>
// CHECK: affine.for %{{.+}} = 0 to 4 {
// CHECK:   affine.load %{{.+}}[{{.+}}] : memref<64x64xf32>
// CHECK:   affine.store %{{.+}}, %{{.+}}[{{.+}}] : memref<16x16xf32>
// CHECK:   affine.load %{{.+}}[{{.+}}] : memref<64x64xf32>
// CHECK:   affine.store %{{.+}}, %{{.+}}[{{.+}}] : memref<16x16xf32>
// CHECK:   "enzymexla.barrier"
// CHECK:   affine.for %{{.+}} = 0 to 16 {
// CHECK:     affine.load %{{.+}}[{{.+}}] : memref<16x16xf32>
// CHECK:     affine.load %{{.+}}[{{.+}}] : memref<16x16xf32>
// CHECK:     affine.load %{{.+}}[] : memref<f32>
// CHECK:     arith.mulf
// CHECK:     arith.addf
// CHECK:     affine.store %{{.+}}, %{{.+}}[] : memref<f32>
// CHECK:   }
// CHECK:   "enzymexla.barrier"
// CHECK: }
// CHECK: affine.load %{{.+}}[] : memref<f32>
// CHECK: affine.store %{{.+}}, %{{.+}}[{{.+}}] : memref<64x64xf32>
