// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

// A launch operand whose type does not match the kernel parameter is coerced.
// An integer and a pointer are not bitcastable, so the integer-to-pointer and
// pointer-to-integer directions need inttoptr and ptrtoint.
module attributes {gpu.container_module} {
  llvm.func @__mlir_cuda_caller_phase3(...)

  llvm.func internal @"reactant$kern"(%arg0: !llvm.ptr) {
    llvm.return
  }

  llvm.func @caller(%int_arg: i64) {
    %f = llvm.mlir.addressof @"reactant$kern" : !llvm.ptr
    %dim = llvm.mlir.constant(1 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%f, %dim, %dim, %dim, %dim, %dim, %dim, %shmem, %stream, %int_arg) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

// CHECK-NOT: llvm.bitcast %{{.*}} : i64 to !llvm.ptr
// CHECK: llvm.inttoptr
