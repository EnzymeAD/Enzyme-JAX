// RUN: enzymexlamlir-opt %s -remove-duplicate-func-def | FileCheck %s

// Outlining emits one gpu.module per launch site, so a kernel launched twice
// (here with its arguments swapped, as in a ping-pong loop) yields identical
// modules that would each get their own binary, stub and registration ctor.

module attributes {gpu.container_module} {
  // CHECK: gpu.module @kern_mod
  gpu.module @kern_mod {
    gpu.func @kern(%arg0: !llvm.ptr, %arg1: !llvm.ptr) kernel {
      %0 = llvm.load %arg0 : !llvm.ptr -> f32
      llvm.store %0, %arg1 : f32, !llvm.ptr
      gpu.return
    }
  }

  // CHECK-NOT: gpu.module @kern_mod_0
  gpu.module @kern_mod_0 {
    gpu.func @kern(%arg0: !llvm.ptr, %arg1: !llvm.ptr) kernel {
      %0 = llvm.load %arg0 : !llvm.ptr -> f32
      llvm.store %0, %arg1 : f32, !llvm.ptr
      gpu.return
    }
  }

  // A module whose body differs must survive.
  // CHECK: gpu.module @other_mod
  gpu.module @other_mod {
    gpu.func @kern(%arg0: !llvm.ptr, %arg1: !llvm.ptr) kernel {
      %0 = llvm.load %arg1 : !llvm.ptr -> f32
      llvm.store %0, %arg0 : f32, !llvm.ptr
      gpu.return
    }
  }

  // CHECK-LABEL: @main
  func.func @main(%src: !llvm.ptr, %dst: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    // CHECK: gpu.launch_func @kern_mod::@kern
    gpu.launch_func @kern_mod::@kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%src : !llvm.ptr, %dst : !llvm.ptr)
    // The duplicate's launch site is repointed at the survivor, keeping its
    // own (swapped) arguments.
    // CHECK: gpu.launch_func @kern_mod::@kern
    // CHECK-SAME: args(%arg1 : !llvm.ptr, %arg0 : !llvm.ptr)
    gpu.launch_func @kern_mod_0::@kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%dst : !llvm.ptr, %src : !llvm.ptr)
    // CHECK: gpu.launch_func @other_mod::@kern
    gpu.launch_func @other_mod::@kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%src : !llvm.ptr, %dst : !llvm.ptr)
    return
  }
}
