// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// parallelize-block-ops moves the ops beside an inner parallel into its body,
// replacing their uses there. The inner parallel's own bounds are not such a
// use: they read the original from outside, and erasing it leaves them
// dangling. This bound is computed from the outer induction variable, so it
// cannot be lifted out of the block either -- the pattern declines the shape
// instead of breaking it.

module {
  func.func @pin(%n: index, %out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c8, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%n) step (%c1) {
        %bound = arith.muli %i, %c8 : index
        %read = memref.load %out[%i] : memref<?xf64>
        scf.parallel (%j) = (%c0) to (%bound) step (%c1) {
          memref.store %read, %out[%j] : memref<?xf64>
          scf.reduce
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// The bound stays beside the inner parallel, which still reads it. That
// parallel cannot give the launch its thread shape -- a trip count computed
// from the grid index is not a block dimension -- so the launch takes one
// thread per block and the parallel is left inside it to be serialized, with
// its bound computation still in front of it.

// CHECK-LABEL: func.func @pin
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %{{.+}}, %{{.+}} = %[[ONE:.+]], %{{.+}} = %[[ONE]]) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[ONE]], %{{.+}} = %[[ONE]], %{{.+}} = %[[ONE]])
// CHECK: %[[BID:.+]] = gpu.block_id x
// CHECK-NEXT: %[[B:.+]] = arith.muli %[[BID]], %{{.+}} : index
// CHECK-NEXT: scf.parallel (%{{.+}}) = (%{{.+}}) to (%[[B]])
// CHECK-NEXT: %[[R:.+]] = memref.load
// CHECK-NEXT: memref.store %[[R]]
