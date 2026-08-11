// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower{wrapParallelOps=true})" | FileCheck %s

// A kernel lives in a gpu.module, so the launch names it with a nested symbol
// reference, which cannot be lowered to a func.call here. The launch is left
// for the passes that lower gpu.launch_func directly.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @kern() kernel {
      gpu.return
    }
  }
  func.func @caller() {
    %c1 = arith.constant 1 : index
    %shmem = arith.constant 0 : i32
    gpu.launch_func @gpum::@kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %shmem
    return
  }
}

// CHECK-LABEL: func.func @caller
// CHECK: gpu.launch_func @gpum::@kern
