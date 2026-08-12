// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm | FileCheck %s

// A byval kernel argument normally arrives as a pointer to the argument's
// copy, which is what the kernel argument array holds. Here the parameter has
// been scalarised and the byval attribute left behind, so the argument arrives
// by value: it needs a temporary and a store like any other argument, rather
// than having its own bits reinterpreted as an address.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @kern(%arg0: i32) kernel {
      gpu.return
    }
  }
  func.func @caller(%v: i32) {
    %c1 = arith.constant 1 : index
    %shmem = arith.constant 0 : i32
    gpu.launch_func @gpum::@kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %shmem args(%v : i32) {reactant.arg_attrs = [{llvm.byval = !llvm.struct<"S", (i8)>}]}
    return
  }
}

// CHECK-LABEL: llvm.func @caller
// CHECK: %[[SLOT:.+]] = llvm.alloca %{{.+}} x i32
// CHECK: llvm.store %{{.+}}, %[[SLOT]]
