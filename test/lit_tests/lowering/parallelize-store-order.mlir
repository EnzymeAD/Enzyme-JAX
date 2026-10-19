// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Write ops moved into the inner parallel land in a thread-zero if, together
// with everything that feeds them, in original program order. The if used to
// collect only the writes while their operands were cloned after it, leaving
// a use above its def when a store's operand was computed between two writes.

module {
  func.func @f(%in: memref<?xf64, 1>, %out: memref<?xf64, 1>, %s: index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f64
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%e) = (%c0) to (%c3) step (%c1) {
        %alloca = memref.alloca() : memref<5xf64>
        memref.store %cst, %alloca[%c0] : memref<5xf64>
        %sum = scf.for %q = %c0 to %c2 step %c1 iter_args(%acc = %cst) -> (f64) {
          %v = memref.load %in[%q] : memref<?xf64, 1>
          %a = arith.addf %acc, %v : f64
          scf.yield %a : f64
        }
        memref.store %sum, %alloca[%c1] : memref<5xf64>
        scf.parallel (%q) = (%c0) to (%c2) step (%c1) {
          %v = memref.load %alloca[%q] : memref<5xf64>
          %i = arith.muli %e, %c2 : index
          %j = arith.addi %i, %q : index
          memref.store %v, %out[%j] : memref<?xf64, 1>
          scf.reduce
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK-SAME: %[[IN:[a-z0-9]+]]: memref<?xf64, 1>, %[[OUT:[a-z0-9]+]]: memref<?xf64, 1>
// CHECK: gpu.launch
// CHECK: %[[BID:[a-z0-9_]+]] = gpu.block_id x
// CHECK-NEXT: %[[ALLOCA:[a-z0-9_]+]] = memref.alloca() : memref<5xf64, 5>
// CHECK-NEXT: %[[BUF:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOCA]] : memref<5xf64, 5> to memref<5xf64>
// CHECK-NEXT: %[[TID:[a-z0-9_]+]] = gpu.thread_id x
// CHECK-NEXT: %[[COND:[a-z0-9_]+]] = arith.cmpi eq, %[[TID]], %[[C0:[a-z0-9_]+]] : index
// CHECK-NEXT: scf.if %[[COND]] {
// CHECK-NEXT: memref.store %[[CST:[a-z0-9_]+]], %[[BUF]][%[[C0]]] : memref<5xf64>
// CHECK-NEXT: %[[SUM:[a-z0-9_]+]] = scf.for %[[Q:[a-z0-9_]+]] = %[[C0]] to %[[C2:[a-z0-9_]+]] step %[[C1:[a-z0-9_]+]] iter_args(%[[ACC:[a-z0-9_]+]] = %[[CST]]) -> (f64) {
// CHECK-NEXT: %[[V:[a-z0-9_]+]] = memref.load %[[IN]][%[[Q]]] : memref<?xf64, 1>
// CHECK-NEXT: %[[ADD:[a-z0-9_]+]] = arith.addf %[[ACC]], %[[V]] : f64
// CHECK-NEXT: scf.yield %[[ADD]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: memref.store %[[SUM]], %[[BUF]][%[[C1]]] : memref<5xf64>
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[L:[a-z0-9_]+]] = memref.load %[[BUF]][%[[TID]]] : memref<5xf64>
// CHECK-NEXT: %[[MUL:[a-z0-9_]+]] = arith.muli %[[BID]], %[[C2]] : index
// CHECK-NEXT: %[[IDX:[a-z0-9_]+]] = arith.addi %[[MUL]], %[[TID]] : index
// CHECK-NEXT: memref.store %[[L]], %[[OUT]][%[[IDX]]] : memref<?xf64, 1>
// CHECK-NEXT: gpu.terminator
