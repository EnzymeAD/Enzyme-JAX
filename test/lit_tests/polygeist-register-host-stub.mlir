// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm -split-input-file | FileCheck %s

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

// -----

// A kernel outlined from a parallel region has no device stub of its own, so a
// synthetic one is registered. It must not bind its enclosing function.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @main_kernel() kernel {
      gpu.return
    }
  }
  llvm.func @main() {
    llvm.return
  }
}

// CHECK-LABEL: llvm.func private @gpum_gpubin_ctor()
// CHECK-NOT: llvm.mlir.addressof @main :
// CHECK: %[[SYNTH:.+]] = llvm.mlir.addressof @__polygeist_gpum_main_kernel_device_stub : !llvm.ptr
// CHECK: RegisterFunction(%{{.+}}, %[[SYNTH]],
// CHECK-NOT: RegisterFunction(%
// CHECK: llvm.func internal @__polygeist_gpum_main_kernel_device_stub()

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
