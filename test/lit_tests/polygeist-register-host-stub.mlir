// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm -split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm="backend=cuda cuda-get-kernel=true" -split-input-file | FileCheck %s --check-prefix=CUDA-GET-KERNEL

// A kernel carrying its recorded host symbol is registered under that
// symbol, which is the address user code can take, rather than under a
// synthetic stub that nothing else refers to.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @"reactant$_Z16__device_stub__kPi"() kernel attributes {"polygeist.host_symbol" = "_Z16__device_stub__kPi"} {
      gpu.return
    }
  }
  llvm.func @_Z16__device_stub__kPi() {
    llvm.return
  }
}

// CHECK-NOT: llvm.func internal @__polygeist_gpum_reactant$_Z16__device_stub__kPi_device_stub
// CHECK-LABEL: llvm.func private @gpum_gpubin_ctor()
// CHECK: %[[ORIG:.+]] = llvm.mlir.addressof @_Z16__device_stub__kPi : !llvm.ptr
// CHECK: RegisterFunction(%{{.+}}, %[[ORIG]],
// CHECK-NOT: RegisterFunction(%
// CHECK-NOT: cudaGetKernel

// CUDA-GET-KERNEL-LABEL: llvm.func private @gpum_gpubin_ctor()
// CUDA-GET-KERNEL-DAG: %[[FUNPTR:.+]] = llvm.mlir.addressof @__polygeist_gpum_reactant$_Z16__device_stub__kPi_fun_ptr : !llvm.ptr
// CUDA-GET-KERNEL-DAG: %[[ORIG:.+]] = llvm.mlir.addressof @_Z16__device_stub__kPi : !llvm.ptr
// CUDA-GET-KERNEL: __cudaRegisterFunction(%[[MOD:.+]], %[[ORIG]],
// CUDA-GET-KERNEL: __cudaRegisterFatBinaryEnd(%[[MOD]])
// CUDA-GET-KERNEL-NEXT: llvm.call @cudaGetKernel(%[[FUNPTR]], %[[ORIG]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CUDA-GET-KERNEL: llvm.mlir.global internal @__polygeist_gpum_reactant$_Z16__device_stub__kPi_fun_ptr() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr

// -----

// A kernel outlined from a parallel region has no device stub of its own, so a
// synthetic one is registered. It must not bind its enclosing function.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @main_kernel() kernel {
      gpu.return
    }
  }
  llvm.func @main() -> !llvm.ptr {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i64 = arith.constant 0 : i64
    gpu.launch_func @gpum::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %c0_i32
    %occ = "enzymexla.gpu_occupancy"(%c64_i32, %c0_i64, %c0_i32) <{fn = @gpum::@main_kernel}> : (i32, i64, i32) -> i32
    %addr = "enzymexla.gpu_kernel_address"() <{fn = @gpum::@main_kernel}> : () -> !llvm.ptr
    llvm.return %addr : !llvm.ptr
  }
}

// CHECK-LABEL: llvm.func private @gpum_gpubin_ctor()
// CHECK-NOT: llvm.mlir.addressof @main :
// CHECK: %[[SYNTH:.+]] = llvm.mlir.addressof @__polygeist_gpum_main_kernel_device_stub : !llvm.ptr
// CHECK: RegisterFunction(%{{.+}}, %[[SYNTH]],
// CHECK-NOT: RegisterFunction(%
// CHECK: llvm.func internal @__polygeist_gpum_main_kernel_device_stub()

// CUDA-GET-KERNEL-LABEL: llvm.func @main() -> !llvm.ptr
// CUDA-GET-KERNEL: %[[FUN_GLOBAL:.+]] = llvm.mlir.addressof @__polygeist_gpum_main_kernel_fun_ptr : !llvm.ptr
// CUDA-GET-KERNEL: %[[LAUNCH_HANDLE:.+]] = llvm.load %[[FUN_GLOBAL]] : !llvm.ptr -> !llvm.ptr
// CUDA-GET-KERNEL: llvm.call @cudaLaunchKernel(%[[LAUNCH_HANDLE]],
// CUDA-GET-KERNEL: %[[OCC_HANDLE:.+]] = llvm.load %[[FUN_GLOBAL]] : !llvm.ptr -> !llvm.ptr
// CUDA-GET-KERNEL: llvm.call @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(%{{.+}}, %[[OCC_HANDLE]],
// CUDA-GET-KERNEL: %[[ADDR_HANDLE:.+]] = llvm.load %[[FUN_GLOBAL]] : !llvm.ptr -> !llvm.ptr
// CUDA-GET-KERNEL: llvm.return %[[ADDR_HANDLE]] : !llvm.ptr
// CUDA-GET-KERNEL-LABEL: llvm.func private @gpum_gpubin_ctor()
// CUDA-GET-KERNEL-DAG: %[[FUNPTR2:.+]] = llvm.mlir.addressof @__polygeist_gpum_main_kernel_fun_ptr : !llvm.ptr
// CUDA-GET-KERNEL-DAG: %[[SYNTH:.+]] = llvm.mlir.addressof @__polygeist_gpum_main_kernel_device_stub : !llvm.ptr
// CUDA-GET-KERNEL: __cudaRegisterFunction(%[[MOD2:.+]], %[[SYNTH]],
// CUDA-GET-KERNEL: __cudaRegisterFatBinaryEnd(%[[MOD2]])
// CUDA-GET-KERNEL-NEXT: llvm.call @cudaGetKernel(%[[FUNPTR2]], %[[SYNTH]]) : (!llvm.ptr, !llvm.ptr) -> i32

// -----

// The recorded host symbol is authoritative even when the kernel's name does
// not follow the stub naming convention.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @renamed_kernel() kernel attributes {"polygeist.host_symbol" = "_Z4funv"} {
      gpu.return
    }
  }
  llvm.func @_Z4funv() {
    llvm.return
  }
}

// CHECK-LABEL: llvm.func private @gpum_gpubin_ctor()
// CHECK: %[[ORIG:.+]] = llvm.mlir.addressof @_Z4funv : !llvm.ptr
// CHECK: RegisterFunction(%{{.+}}, %[[ORIG]],
// CHECK-NOT: RegisterFunction(%

// CUDA-GET-KERNEL-LABEL: llvm.func private @gpum_gpubin_ctor()
// CUDA-GET-KERNEL-DAG: %[[FUNPTR3:.+]] = llvm.mlir.addressof @__polygeist_gpum_renamed_kernel_fun_ptr : !llvm.ptr
// CUDA-GET-KERNEL-DAG: %[[ORIG:.+]] = llvm.mlir.addressof @_Z4funv : !llvm.ptr
// CUDA-GET-KERNEL: __cudaRegisterFunction(%[[MOD3:.+]], %[[ORIG]],
// CUDA-GET-KERNEL: __cudaRegisterFatBinaryEnd(%[[MOD3]])
// CUDA-GET-KERNEL-NEXT: llvm.call @cudaGetKernel(%[[FUNPTR3]], %[[ORIG]]) : (!llvm.ptr, !llvm.ptr) -> i32
