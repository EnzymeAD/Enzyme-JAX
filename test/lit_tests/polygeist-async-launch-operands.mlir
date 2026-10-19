// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm | FileCheck %s

// An async launch carries its stream after the kernel operands, so the kernel
// arguments are not the trailing operands. Both arguments have to reach the
// argument array, and the stream must not be mistaken for one of them.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @kern(%arg0: i32, %arg1: i64) kernel {
      gpu.return
    }
  }
  func.func @caller(%a: i32, %b: i64, %stream: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %shmem = arith.constant 0 : i32
    gpu.launch_func <%stream : !llvm.ptr> @gpum::@kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %shmem args(%a : i32, %b : i64)
    return
  }
}

// CHECK-LABEL: llvm.func @caller
// CHECK-DAG: llvm.alloca %{{.+}} x i32
// CHECK-DAG: llvm.alloca %{{.+}} x i64
