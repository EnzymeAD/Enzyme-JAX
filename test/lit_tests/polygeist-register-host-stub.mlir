// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm -split-input-file | FileCheck %s

// Kernel addresses that could not be rewritten statically reach the runtime as
// the original, un-prefixed host stub, so it is registered against the same
// device function alongside the synthetic stub.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @"reactant$_Z16__device_stub__kPi"() kernel {
      gpu.return
    }
  }
  llvm.func @_Z16__device_stub__kPi() {
    llvm.return
  }
}

// CHECK-LABEL: llvm.func private @gpum_gpubin_ctor()
// CHECK-DAG: %[[ORIG:.+]] = llvm.mlir.addressof @_Z16__device_stub__kPi : !llvm.ptr
// CHECK-DAG: %[[SYNTH:.+]] = llvm.mlir.addressof @__polygeist_gpum_reactant$_Z16__device_stub__kPi_device_stub : !llvm.ptr
// CHECK: RegisterFunction(%{{.+}}, %[[SYNTH]],
// CHECK: RegisterFunction(%{{.+}}, %[[ORIG]],

// -----

// An outlined parallel region is not a device stub and must not bind its
// enclosing function.
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
// CHECK: RegisterFunction
// CHECK-NOT: RegisterFunction
