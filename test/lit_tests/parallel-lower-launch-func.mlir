// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower{wrapParallelOps=true})" | FileCheck %s

// A kernel lives in a gpu.module, so the launch names it with a nested symbol
// reference that func.call cannot carry. Launch recognition clones the
// kernel into the gpu.module under its own name, so the top-level original
// shares the leaf name and body: lower the launch through a call to that.
// The launch carries i64 dims, which become index for the parallel loops.
module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @kern(%arg0: memref<?xf32>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.0 : f32
      memref.store %cst, %arg0[%c0] : memref<?xf32>
      gpu.return
    }
  }
  func.func @kern(%arg0: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    memref.store %cst, %arg0[%c0] : memref<?xf32>
    return
  }
  func.func @caller(%arg0: memref<?xf32>, %n: i64) {
    %c1 = arith.constant 1 : i64
    %shmem = arith.constant 0 : i32
    gpu.launch_func @gpum::@kern blocks in (%n, %c1, %c1) threads in (%c1, %c1, %c1) : i64 dynamic_shared_memory_size %shmem args(%arg0 : memref<?xf32>)
    return
  }
}

// CHECK-LABEL: func.func @caller
// CHECK-SAME: (%[[BUF:.+]]: memref<?xf32>, %[[N:.+]]: i64)
// CHECK-DAG: %[[NI:.+]] = arith.index_cast %[[N]] : i64 to index
// CHECK-DAG: %[[ONE:.+]] = arith.constant 1 : index
// CHECK: "enzymexla.gpu_wrapper"(%[[NI]], %[[ONE]], %[[ONE]], %[[ONE]], %[[ONE]], %[[ONE]])
// CHECK: scf.parallel (%{{.*}}) = (%{{.*}}) to (%[[NI]], %[[ONE]], %[[ONE]])
// CHECK: scf.parallel
// CHECK: memref.store %{{.*}}, %[[BUF]][%{{.*}}] : memref<?xf32>
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: func.call
