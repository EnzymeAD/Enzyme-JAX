// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition{backend=rocm})" | FileCheck %s

// The kernel was compiled for gfx950. Its address escapes to a runtime query,
// so it is promoted with gpu.launch_func, and the gpu.module it lands in must
// target that chip and the kernel's features rather than a hardcoded default.
module attributes {gpu.container_module} {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @cudaFuncGetAttributes(!llvm.ptr, !llvm.ptr) -> i32

  llvm.func internal @kern() attributes {passthrough = [["target-cpu", "gfx950"]], target_cpu = "gfx950", target_features = #llvm.target_features<["+dpp", "+wavefrontsize64"]>} {
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

// CHECK: gpu.module @__mlir_gpu_module [#rocdl.target<O = 3, chip = "gfx950", features = "+dpp,+wavefrontsize64">]
// CHECK: gpu.func @kern() kernel
// CHECK: gpu.launch_func
// CHECK-SAME: @kern
