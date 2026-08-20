// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

// Incorrect address spaces should not cause the compiler to crash
module attributes {gpu.container_module} {
  func.func @hostfunc(%iv: index, %jv: index, %ptr: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %unused = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf32>
    // the result of gpu.alloc should have addrspace 1, assuming global memory
    %alloc = gpu.alloc (%c1) : memref<?xf32>
    gpu.launch_func @gpumod::@gpufunc blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<?xf32>)
    return
  }

  gpu.module @gpumod [#nvvm.target<O = 3, chip = "sm_120", features = "+ptx73", flags = {}>] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} {
    gpu.func @gpufunc(%arg0: memref<?xf32>) kernel {
      %thread_id_x = gpu.thread_id x
      %cst = arith.constant 1.0 : f32
      memref.store %cst, %arg0[%thread_id_x] : memref<?xf32>
      gpu.return
    }
  }
}

// CHECK-LABEL:   llvm.func @hostfunc(
// CHECK-SAME:      %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64,
// CHECK-SAME:      %[[ARG2:.*]]: !llvm.ptr) {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[MLIR_1:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:           %[[MLIR_2:.*]] = llvm.mlir.addressof @__polygeist_gpumod_gpufunc_device_stub : !llvm.ptr
// CHECK:           %[[MLIR_3:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[MLIR_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[MLIR_5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[ALLOCA_0:.*]] = llvm.alloca %[[MLIR_6]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           %[[ALLOCA_1:.*]] = llvm.alloca %[[MLIR_6]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           %[[ALLOCA_2:.*]] = llvm.alloca %[[MLIR_5]] x !llvm.ptr<1> : (i64) -> !llvm.ptr
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[MLIR_3]][1] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           %[[PTRTOINT_0:.*]] = llvm.ptrtoint %[[GETELEMENTPTR_0]] : !llvm.ptr to i64
// CHECK:           %[[CALL_0:.*]] = llvm.call @cudaMalloc(%[[ALLOCA_2]], %[[PTRTOINT_0]]) : (!llvm.ptr, i64) -> i32
// CHECK:           %[[LOAD_0:.*]] = llvm.load %[[ALLOCA_2]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK:           %[[ADDRSPACECAST_0:.*]] = llvm.addrspacecast %[[LOAD_0]] : !llvm.ptr<1> to !llvm.ptr
// CHECK:           llvm.store %[[ADDRSPACECAST_0]], %[[ALLOCA_0]] : !llvm.ptr, !llvm.ptr
