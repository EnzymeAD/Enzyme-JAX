// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// A runtime-bound accumulation loop raised to a parallel-with-reduction sits
// directly under the single parallel left when grid and block were fused. It
// can never be a thread parallel -- a launch has no reduction semantics -- so
// it must be serialized like any inner loop. Leaving it in place sent it into
// patterns that only handle resultless parallels and crashed the compiler.

module {
  func.func @f(%in: memref<?xf64, 1>, %out: memref<?xf64, 1>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f64
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%e) = (%c0) to (%c3) step (%c1) {
        %sum = scf.parallel (%q) = (%c0) to (%c2) step (%c1) init (%cst) -> f64 {
          %v = memref.load %in[%q] : memref<?xf64, 1>
          scf.reduce(%v : f64) {
          ^bb0(%a: f64, %b: f64):
            %s = arith.addf %a, %b : f64
            scf.reduce.return %s : f64
          }
        }
        memref.store %sum, %out[%e] : memref<?xf64, 1>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK: gpu.launch
// CHECK: %[[ACC:.+]] = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[IT:.+]] = %{{.+}}) -> (f64)
// CHECK: %[[V:.+]] = memref.load
// CHECK: %[[S:.+]] = arith.addf %[[IT]], %[[V]]
// CHECK: scf.yield %[[S]]
// CHECK: memref.store %[[ACC]]
