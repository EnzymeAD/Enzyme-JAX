// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

// Test that cudaMalloc calls are properly recognized and replaced with gpu.alloc
// operations by the GPULaunchRecognition pass.

module {
  llvm.func @test_cudamalloc() -> i32 {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %size = llvm.mlir.constant(1024 : i64) : i64
    %devptr = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %ret = llvm.call @cudaMalloc(%devptr, %size) : (!llvm.ptr, i64) -> (i32)
    llvm.return %ret : i32
  }
  llvm.func @cudaMalloc(!llvm.ptr, i64) -> i32 attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @test_cudamalloc
// CHECK:         %[[SIZE:.*]] = arith.index_cast {{.*}} : i64 to index
// CHECK:         %[[MEMREF:.*]] = gpu.alloc  (%[[SIZE]]) : memref<?xi8, 1>
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[MEMREF]]) : (memref<?xi8, 1>) -> !llvm.ptr
// CHECK:         llvm.store %[[PTR]], {{.*}} : !llvm.ptr, !llvm.ptr
// CHECK:         llvm.mlir.zero : i32
// CHECK-NOT:     llvm.call @cudaMalloc