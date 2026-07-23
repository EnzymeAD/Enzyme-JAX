// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  func.func @foo(%iv: index, %jv: index, %ptr: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %unused = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf32>
    %alloc = gpu.alloc (%c1) : memref<?xf32, 1>
    gpu.launch_func @gpumod::@gpufunc blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<?xf32, 1>)
    gpu.dealloc %alloc : memref<?xf32, 1>
    return
  }

  gpu.module @gpumod [#nvvm.target<O = 3, chip = "sm_120", features = "+ptx73", flags = {}>] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} {
    gpu.func @gpufunc(%arg0: memref<?xf32, 1>) kernel {
      %thread_id_x = gpu.thread_id x
      // memref.alloc() ops inside gpu.modules should lower to malloc calls where the
      // malloc is declared inside the gpu.module
      %tmp = memref.alloc() : memref<f32>
      %ld = memref.load %arg0[%thread_id_x] : memref<?xf32, 1>
      memref.store %ld, %tmp[] : memref<f32>
      %ld1 = memref.load %tmp[] : memref<f32>
      memref.store %ld1, %arg0[%thread_id_x] : memref<?xf32, 1>
      memref.dealloc %tmp : memref<f32>
      gpu.return
    }
  }
}

// Ensure the malloc/free declarations are inside the gpu.module
// CHECK-LABEL: gpu.module @gpumod
// CHECK: llvm.func @free(!llvm.ptr)
// CHECK: llvm.func @malloc(i64) -> !llvm.ptr
