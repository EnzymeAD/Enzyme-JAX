// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @mod [#nvvm.target] {
    gpu.func @bounded(%arg0: f32) kernel attributes {known_block_size = array<i32: 8, 8, 8>} {
      gpu.return
    }
    gpu.func @unbounded(%arg0: f32) kernel {
      gpu.return
    }
  }
}

// CHECK: llvm.func @bounded({{.*}}known_block_size = array<i32: 8, 8, 8>, nvvm.kernel, nvvm.maxntid = array<i32: 8, 8, 8>
// CHECK: llvm.func @unbounded(
// CHECK-NOT: @unbounded{{.*}}nvvm.maxntid
