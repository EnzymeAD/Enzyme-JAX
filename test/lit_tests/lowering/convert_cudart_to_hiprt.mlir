// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-cudart-to-hiprt)" | FileCheck %s

module {
  llvm.func @cudaMalloc(!llvm.ptr, i64) -> i32
  llvm.func @cudaFree(!llvm.ptr) -> i32
  llvm.func @cudaMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
  llvm.func @cudaDeviceSynchronize() -> i32
  llvm.func @cudaMemset(!llvm.ptr, i32, i64) -> i32
  llvm.func @cudaGetLastError() -> i32
  llvm.func @someOtherFunction(!llvm.ptr) -> i32
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.func @test_llvm_cuda_calls(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32

    %0 = llvm.call @cudaMalloc(%arg0, %arg2) : (!llvm.ptr, i64) -> i32
    %1 = llvm.call @cudaMemcpy(%arg0, %arg1, %arg2, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
    %2 = llvm.call @cudaMemset(%arg0, %c0, %arg2) : (!llvm.ptr, i32, i64) -> i32
    %3 = llvm.call @cudaDeviceSynchronize() : () -> i32
    %4 = llvm.call @cudaFree(%arg0) : (!llvm.ptr) -> i32
    %5 = llvm.call @cudaGetLastError() : () -> i32

    llvm.return %c0 : i32
  }

  llvm.func @test_nvvm_barrier_conversion(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.store %0, %arg0 : i32, !llvm.ptr
    nvvm.barrier0
    %1 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.return
  }
}

// CHECK-DAG: llvm.func @hipMalloc(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @hipFree(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @hipMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK-DAG: llvm.func @hipDeviceSynchronize() -> i32
// CHECK-DAG: llvm.func @hipMemset(!llvm.ptr, i32, i64) -> i32
// CHECK-DAG: llvm.func @hipGetLastError() -> i32

// Check non-CUDA functions remain unchanged
// CHECK-DAG: llvm.func @someOtherFunction(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @printf(!llvm.ptr, ...) -> i32

// CHECK-LABEL: llvm.func @test_llvm_cuda_calls
// CHECK: llvm.call @hipMalloc
// CHECK: llvm.call @hipMemcpy
// CHECK: llvm.call @hipMemset
// CHECK: llvm.call @hipDeviceSynchronize
// CHECK: llvm.call @hipFree
// CHECK: llvm.call @hipGetLastError

// CHECK-LABEL: llvm.func @test_nvvm_barrier_conversion
// CHECK: llvm.store
// CHECK: rocdl.barrier
// CHECK: llvm.load
