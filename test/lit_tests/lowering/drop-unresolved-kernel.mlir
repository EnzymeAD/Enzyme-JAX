// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" --verify-diagnostics | FileCheck %s

// A wrapper still carrying a host target holds a kernel that was never
// resolved to a device body: its content is a call to an external device
// stub, and an outlined module would inherit the host chip and fail ptxas.
// Even when the parallel's shape matches the wrapper's launch bounds -- it
// always does, both were built from the same launch configuration -- the
// kernel must be dropped, with an error, not emitted.

module {
  llvm.func @"reactant$stub"(i32)
  func.func @f() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c24 = arith.constant 24 : index
    // expected-error@below {{kernel with host target 'x86-64' was not resolved to a device body; the kernel is dropped}}
    %r = "enzymexla.gpu_wrapper"(%c3, %c1, %c1, %c24, %c24, %c1) ({
      scf.parallel (%bx, %tx, %ty) = (%c0, %c0, %c0) to (%c3, %c24, %c24) step (%c1, %c1, %c1) {
        %i = arith.index_cast %tx : index to i32
        llvm.call @"reactant$stub"(%i) : (i32) -> ()
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) {target_cpu = "x86-64"} : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK: %[[ERR:.+]] = arith.constant 701
// CHECK-NOT: gpu.launch
// CHECK-NOT: enzymexla.gpu_wrapper
// CHECK: return %[[ERR]]
