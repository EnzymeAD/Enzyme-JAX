// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// The wrapper's result is the launch error code. On the path where the kernel
// splits successfully the wrapper was erased without replacing that result, so
// any remaining use was left dangling and the next fold crashed the compiler.

module {
  func.func @f(%m: memref<?xf64, 1>, %v: f64) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %r = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%c256) step (%c1) {
        %g = arith.cmpi sle, %i, %c2 : index
        scf.if %g {
          memref.store %v, %m[%i] : memref<?xf64, 1>
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %r : index
  }
}

// CHECK-LABEL: func.func @f(
// CHECK: %[[Z:.+]] = arith.constant 0 : index
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch
// CHECK: return %[[Z]] : index
