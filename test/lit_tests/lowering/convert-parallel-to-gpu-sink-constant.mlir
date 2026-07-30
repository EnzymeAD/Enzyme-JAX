// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Test that convert-parallel-to-gpu1 inserts sunk constants at the start of
// launchOpBody so they dominate all uses in the kernel.

module {
  func.func @sink_constant_kernel(%A: memref<?xf32, 1>) {
    %cst = arith.constant 4.200000e+01 : f32
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %wrapper_result = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
      scf.parallel (%i) = (%c1) to (%c100) step (%c1) {
        memref.store %cst, %A[%i] : memref<?xf32, 1>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @sink_constant_kernel
// CHECK:       gpu.launch
// CHECK:         %[[CST:.*]] = arith.constant 4.200000e+01 : f32
// CHECK:         memref.store %[[CST]], %{{.*}}[%{{.*}}] : memref<?xf32, 1>
