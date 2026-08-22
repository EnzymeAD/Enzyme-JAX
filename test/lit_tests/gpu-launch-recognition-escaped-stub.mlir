// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

// The kernel address user code holds is that of clang's host stub. Here it
// escapes into a call rather than reaching a runtime query directly, so
// nothing rewrites it to the device symbol. The kernel is still captured and
// has to reach a gpu.module, or it is never emitted and never registered.
module attributes {gpu.container_module} {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @escape(!llvm.ptr)

  llvm.func internal @"reactant$_Z16__device_stub__kPi"(%arg0: !llvm.ptr {llvm.nonnull}) attributes {passthrough = [["polygeist.host_symbol", "_Z16__device_stub__kPi"]]} {
    llvm.return
  }

  llvm.func internal @_Z16__device_stub__kPi(%p: !llvm.ptr) {
    %f = llvm.mlir.addressof @"reactant$_Z16__device_stub__kPi" : !llvm.ptr
    %dim = llvm.mlir.constant(1 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%f, %dim, %dim, %dim, %dim, %dim, %dim, %shmem, %stream, %p) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @user() {
    %h = llvm.mlir.addressof @_Z16__device_stub__kPi : !llvm.ptr
    llvm.call @escape(%h) : (!llvm.ptr) -> ()
    llvm.return
  }
}

// CHECK: gpu.func @reactant$_Z16__device_stub__kPi(%{{.+}}: !llvm.ptr {{.*}}) kernel
