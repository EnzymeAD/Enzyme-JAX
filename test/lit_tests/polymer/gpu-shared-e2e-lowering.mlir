// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-polyhedral-opt{block-size-x=16 block-size-y=16 tile-size=16 lower-to-scf=true},convert-parallel-to-gpu1,gpu-kernel-outlining,convert-parallel-to-gpu2{backend=cuda})" | FileCheck %s

module {
  func.func @matmul_e2e(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
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

// CHECK-LABEL: func.func @matmul_e2e
// CHECK: gpu.launch_func
// CHECK: gpu.module
// CHECK-DAG: memref.global {{.*}}@shared_mem_{{[0-9]+}} : memref<16x16xf32, 3>
// CHECK-DAG: memref.global {{.*}}@shared_mem_{{[0-9]+}} : memref<16x16xf32, 3>
// CHECK: gpu.func @{{.+}} kernel
// CHECK: gpu.thread_id
// CHECK: gpu.barrier
