// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Promoting the launch-preamble alloca to memory space 5 cannot hand its
// users a memref.cast back to the space-0 type: memref.cast cannot change
// the memory space. That is what memref.memory_space_cast is for.

module {
  func.func @shmem(%n: index, %out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %w = "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c64, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%n) step (%c1) {
        %sh = memref.alloca() : memref<64xf64>
        %cast = memref.cast %sh : memref<64xf64> to memref<?xf64>
        scf.parallel (%j) = (%c0) to (%c64) step (%c1) {
          memref.store %v, %cast[%j] : memref<?xf64>
          "enzymexla.barrier"(%j) : (index) -> ()
          %x = memref.load %cast[%j] : memref<?xf64>
          memref.store %x, %out[%j] : memref<?xf64>
          scf.reduce
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @shmem
// CHECK: gpu.launch
// CHECK: %[[SH:.+]] = memref.alloca() : memref<64xf64, 5>
// CHECK-NEXT: %[[MR:.+]] = memref.memory_space_cast %[[SH]] : memref<64xf64, 5> to memref<64xf64>
// CHECK: memref.store %{{.+}}, %[[MR]][%{{.+}}] : memref<64xf64>
// CHECK: gpu.barrier
// CHECK: memref.load %[[MR]][%{{.+}}] : memref<64xf64>
