// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

// A kernel whose address escapes to a runtime query is captured, so it is
// promoted with gpu.launch_func rather than lowered to a parallel region. If it
// also has a launch to rewrite and carries no argument attributes -- taking no
// arguments at all is the simplest way to get there -- the attribute list is
// absent rather than empty.
module attributes {gpu.container_module} {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @cudaFuncGetAttributes(!llvm.ptr, !llvm.ptr) -> i32

  llvm.func internal @kern() {
    llvm.return
  }

  llvm.func @caller(%attrs: !llvm.ptr) {
    %f = llvm.mlir.addressof @kern : !llvm.ptr
    %dim = llvm.mlir.constant(1 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%f, %dim, %dim, %dim, %dim, %dim, %dim, %shmem, %stream) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr) -> ()
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %f) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
}

// CHECK: gpu.func @kern() kernel
// CHECK: gpu.launch_func
// CHECK-SAME: @kern
// CHECK-SAME: reactant.arg_attrs = []
