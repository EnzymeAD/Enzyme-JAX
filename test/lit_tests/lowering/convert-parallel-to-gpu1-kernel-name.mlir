// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1,gpu-kernel-outlining)" | FileCheck %s

// A kernel is named after the function it is outlined from, so a function
// whose name is not a valid PTX identifier, e.g. one of Julia's, names its
// launches. The others keep the default name.

module {
  func.func @"Const{typeof(f)}_jit_region"(%out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c32, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%c32) step (%c1) {
        memref.store %v, %out[%i] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }

  func.func @legal(%out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c32, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%c32) step (%c1) {
        memref.store %v, %out[%i] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @"Const{typeof(f)}_jit_region"
// CHECK: gpu.launch_func @Const_typeof_f___jit_region_kernel::@Const_typeof_f___jit_region_kernel
// CHECK: gpu.module @Const_typeof_f___jit_region_kernel
// CHECK-NEXT: gpu.func @Const_typeof_f___jit_region_kernel(
// CHECK-LABEL: func.func @legal
// CHECK: gpu.launch_func @legal_kernel::@legal_kernel
// CHECK: gpu.module @legal_kernel
// CHECK-NEXT: gpu.func @legal_kernel(
