// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1,func.func(gpu-eliminate-barriers))" | FileCheck %s

// Verify that running gpu-eliminate-barriers immediately after convert-parallel-to-gpu1
// (#2893) removes unnecessary barriers introduced by ParallelizeBlockOps (#2883)
// when pre- and post-barrier memory accesses do not conflict, while retaining
// barriers when conflicting shared-memory accesses are present.

module {
  func.func @disjoint_shmem(%in: memref<?xf64, 1>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%e) = (%c0) to (%c3) step (%c1) {
        %a0 = memref.alloca() : memref<4xf64>
        %a1 = memref.alloca() : memref<4xf64>
        scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
          %v = memref.load %in[%q] : memref<?xf64, 1>
          memref.store %v, %a0[%q] : memref<4xf64>
          scf.reduce
        }
        %x = memref.load %in[%c0] : memref<?xf64, 1>
        memref.store %x, %a1[%c0] : memref<4xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }

  func.func @conflicting_shmem(%in: memref<?xf64, 1>, %out: memref<?xf64, 1>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%e) = (%c0) to (%c3) step (%c1) {
        %a0 = memref.alloca() : memref<4xf64>
        scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
          %v = memref.load %in[%q] : memref<?xf64, 1>
          memref.store %v, %a0[%q] : memref<4xf64>
          scf.reduce
        }
        %x = memref.load %a0[%c0] : memref<4xf64>
        memref.store %x, %out[%e] : memref<?xf64, 1>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @disjoint_shmem(
// CHECK: gpu.launch
// CHECK-NOT: gpu.barrier
// CHECK: gpu.terminator

// CHECK-LABEL: func.func @conflicting_shmem(
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK-NEXT: gpu.barrier
// CHECK: gpu.terminator
