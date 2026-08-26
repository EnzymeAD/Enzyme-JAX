// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kern {
    gpu.func @kern(%out: memref<?xf64, 1>) kernel {
      %c0 = arith.constant 0 : index
      %plain = memref.alloca() : memref<2xi32, 5>
      %aligned = memref.alloca() {alignment = 8 : i64} : memref<3xf64, 5>
      %over = memref.alloca() {alignment = 32 : i64} : memref<2x8xf64, 5>
      %one = llvm.mlir.constant(1 : i64) : i64
      %raw = llvm.alloca %one x !llvm.array<4 x f64> {alignment = 16 : i64} : (i64) -> !llvm.ptr<5>
      %raw2 = llvm.alloca %one x !llvm.array<2 x f64> : (i64) -> !llvm.ptr<5>
      %v0 = memref.load %plain[%c0] : memref<2xi32, 5>
      %f0 = arith.sitofp %v0 : i32 to f64
      %v1 = memref.load %aligned[%c0] : memref<3xf64, 5>
      %v2 = memref.load %over[%c0, %c0] : memref<2x8xf64, 5>
      %p = llvm.getelementptr %raw[0, 0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, !llvm.array<4 x f64>
      %v3 = llvm.load %p : !llvm.ptr<5> -> f64
      %p2 = llvm.getelementptr %raw2[0, 0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, !llvm.array<2 x f64>
      %v4 = llvm.load %p2 : !llvm.ptr<5> -> f64
      %s0 = arith.addf %f0, %v1 : f64
      %s1 = arith.addf %s0, %v2 : f64
      %s2 = arith.addf %s1, %v3 : f64
      %s3 = arith.addf %s2, %v4 : f64
      memref.store %s3, %out[%c0] : memref<?xf64, 1>
      gpu.return
    }
  }
}

// CHECK-DAG: memref.global @shared_mem_{{[0-9]+}} : memref<2xi32, 3> = uninitialized{{$}}
// CHECK-DAG: memref.global @shared_mem_{{[0-9]+}} : memref<3xf64, 3> = uninitialized {alignment = 8 : i64}
// CHECK-DAG: memref.global @shared_mem_{{[0-9]+}} : memref<2x8xf64, 3> = uninitialized {alignment = 32 : i64}
// CHECK-DAG: llvm.mlir.global internal @shared_mem_{{[0-9]+}}() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<4 x f64>
// CHECK-DAG: llvm.mlir.global internal @shared_mem_{{[0-9]+}}() {addr_space = 3 : i32} : !llvm.array<2 x f64>
