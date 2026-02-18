// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=rocm})" | FileCheck %s

module attributes {gpu.container_module} {
  llvm.func @main() -> (i32 {llvm.noundef}) attributes {dso_local} {
    %c0_i32 = arith.constant 0 : i32
    llvm.return %c0_i32 : i32
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<?xf64, 1>, %arg1: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      gpu.return
    }
  }
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel(%arg0: memref<?xf64, 1>, %arg1: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      gpu.return
    }
  }
  gpu.module @main_kernel_2 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      gpu.return
    }
  }
}

// CHECK-LABEL: @main

// CHECK: gpu.module @main_kernel [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_0 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_2 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}