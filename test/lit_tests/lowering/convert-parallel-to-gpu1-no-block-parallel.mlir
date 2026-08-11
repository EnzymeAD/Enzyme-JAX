// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// MFEM's CuKernel3D walks a grid-stride loop and does its work in loops
// within it: 'for (k = blockIdx.x; k < N; k += gridDim.x) { body(k); }'. The
// thread-level parallelism the raising finds inside that loop is not a
// parallel directly under the grid parallel, which is all the launch used to
// accept -- and a wrapper it declines does not stay a wrapper, it reaches the
// LLVM lowering, which has no pattern for it at all.
//
// Decide the launch from what there is, and leave what is inside to the
// serialization every loop in a kernel gets anyway.

module {
  func.func @grid_stride(%n: index, %stride: index, %out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c288, %c1, %c1) ({
      scf.parallel (%b) = (%c0) to (%n) step (%c1) {
        %in = arith.cmpi slt, %b, %n : index
        scf.if %in {
          scf.for %k = %b to %n step %stride {
            scf.parallel (%i) = (%c0) to (%c288) step (%c1) {
              memref.store %v, %out[%i] : memref<?xf64>
              scf.reduce
            }
          }
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// The kernel becomes a launch rather than surviving as a wrapper, the guard
// still wraps the whole loop, and the work inside is left to be serialized.
// CHECK-LABEL: func.func @grid_stride
// CHECK-NOT: enzymexla.gpu_wrapper
// CHECK: gpu.launch blocks
// CHECK: scf.if
// CHECK: scf.for
// CHECK: scf.parallel
