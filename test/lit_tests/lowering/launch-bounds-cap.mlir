// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{emitGPUKernelLaunchBounds=true backend=cuda})" | FileCheck %s

// A kernel launched with runtime block sizes must stay launchable at every
// legal size: without a bound ptxas allocates registers as if the block were
// small, and a larger launch fails at runtime with too-many-resources. The
// cap is the architecture maximum. A kernel with constant launch sizes gets
// them as its bound.

module attributes {gpu.container_module} {
  gpu.module @mod [#nvvm.target] {
    gpu.func @dyn() kernel {
      gpu.return
    }
    gpu.func @stat() kernel {
      gpu.return
    }
  }
  func.func @launch(%n: index) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    gpu.launch_func @mod::@dyn blocks in (%c1, %c1, %c1) threads in (%n, %n, %n)
    gpu.launch_func @mod::@stat blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c64)
    return
  }
}

// CHECK: gpu.func @dyn() kernel attributes {nvvm.maxntid = array<i32: 1024, 1, 1>, rocdl.max_flat_work_group_size = 1024 : index}
// CHECK: gpu.func @stat() kernel attributes {nvvm.maxntid = array<i32: 64, 1, 64>, rocdl.max_flat_work_group_size = 4096 : index}
