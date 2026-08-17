// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Parallels nested deeper than the grid and block levels -- here inside an
// if inside the thread parallel -- never map onto the launch: they run,
// serialized, inside a single thread. ParallelizeBlockOps must decline
// them: the ops beside them already run in the right order there, and
// moving them in behind a first-iteration guard both reorders code around
// the sibling parallel's body and manufactures barriers that the
// serialized loop would have to be distributed around.

module {
  func.func @f(%in: memref<?xf64, 1>, %out: memref<?xf64, 1>, %out2: memref<?xf64, 1>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f64
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%t) = (%c0) to (%c256) step (%c1) {
        %alloca = memref.alloca() : memref<4xf64>
        %g = arith.cmpi slt, %t, %c3 : index
        scf.if %g {
          memref.store %cst, %alloca[%c0] : memref<4xf64>
          scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
            %v = memref.load %in[%q] : memref<?xf64, 1>
            memref.store %v, %alloca[%q] : memref<4xf64>
            scf.reduce
          }
          %m = memref.load %alloca[%c0] : memref<4xf64>
          memref.store %m, %out2[%t] : memref<?xf64, 1>
          scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
            %v = memref.load %alloca[%q] : memref<4xf64>
            %i = arith.muli %t, %c2 : index
            %j = arith.addi %i, %q : index
            memref.store %v, %out[%j] : memref<?xf64, 1>
            scf.reduce
          }
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK-SAME: %[[IN:[a-z0-9]+]]: memref<?xf64, 1>, %[[OUT:[a-z0-9]+]]: memref<?xf64, 1>, %[[OUT2:[a-z0-9]+]]: memref<?xf64, 1>
// CHECK: gpu.launch
// CHECK-NOT: gpu.barrier
// CHECK-NOT: enzymexla.barrier
// CHECK: %[[A:[a-z0-9_]+]] = memref.alloca() : memref<4xf64>
// CHECK-NEXT: %[[G:[a-z0-9_]+]] = arith.cmpi slt, %[[T:[a-z0-9_]+]], %{{.+}} : index
// CHECK-NEXT: scf.if %[[G]] {
// CHECK-NEXT: memref.store %[[CST:[a-z0-9_]+]], %[[A]][%[[C0:[a-z0-9_]+]]] : memref<4xf64>
// CHECK-NEXT: scf.parallel (%[[Q1:[a-z0-9_]+]]) = (%[[C0]]) to (%[[C2:[a-z0-9_]+]]) step (%[[C1:[a-z0-9_]+]]) {
// CHECK-NEXT: %[[V1:[a-z0-9_]+]] = memref.load %[[IN]][%[[Q1]]] : memref<?xf64, 1>
// CHECK-NEXT: memref.store %[[V1]], %[[A]][%[[Q1]]] : memref<4xf64>
// CHECK-NEXT: scf.reduce
// CHECK-NEXT: }
// CHECK-NEXT: %[[M:[a-z0-9_]+]] = memref.load %[[A]][%[[C0]]] : memref<4xf64>
// CHECK-NEXT: memref.store %[[M]], %[[OUT2]][%[[T]]] : memref<?xf64, 1>
// CHECK-NEXT: scf.parallel (%[[Q2:[a-z0-9_]+]]) = (%[[C0]]) to (%[[C2]]) step (%[[C1]]) {
// CHECK-NEXT: %[[V2:[a-z0-9_]+]] = memref.load %[[A]][%[[Q2]]] : memref<4xf64>
// CHECK-NEXT: %[[MUL:[a-z0-9_]+]] = arith.muli %[[T]], %[[C2]] : index
// CHECK-NEXT: %[[J:[a-z0-9_]+]] = arith.addi %[[MUL]], %[[Q2]] : index
// CHECK-NEXT: memref.store %[[V2]], %[[OUT]][%[[J]]] : memref<?xf64, 1>
// CHECK-NEXT: scf.reduce
// CHECK-NEXT: }
// CHECK-NOT: gpu.barrier
// CHECK-NOT: enzymexla.barrier
