// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower{wrapParallelOps=true})" --verify-each=0 | FileCheck %s

// gpu.launch_func takes i32 or i64 dimensions (launch recognition passes the
// sign-extended dim3 fields of a kernel whose address escapes, as MFEM's dfem
// kernels do); the gpu.launch and enzymexla.gpu_wrapper it is lowered to take
// index.

module attributes {gpu.container_module} {
  gpu.module @__mlir_gpu_module {
    gpu.func @kern(%p: !llvm.ptr) kernel {
      %c = llvm.mlir.constant(1.0 : f64) : f64
      llvm.store %c, %p : f64, !llvm.ptr
      gpu.return
    }
  }
  func.func @host(%p: !llvm.ptr, %n: i32) {
    %g = llvm.sext %n : i32 to i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %t = llvm.mlir.constant(256 : i64) : i64
    gpu.launch_func @__mlir_gpu_module::@kern blocks in (%g, %one, %one) threads in (%t, %one, %one) : i64 args(%p : !llvm.ptr)
    return
  }
}

// CHECK:      %[[C256:.+]] = "arith.constant"() <{value = 256 : index}> : () -> index
// CHECK-NEXT: %[[C1:.+]] = "arith.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %[[C0:.+]] = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[EXT:.+]] = "llvm.sext"(%arg1) : (i32) -> i64
// CHECK-NEXT: %[[GRID:.+]] = "arith.index_cast"(%[[EXT]]) : (i64) -> index
// CHECK-NEXT: %{{.+}} = "enzymexla.gpu_wrapper"(%[[GRID]], %[[C1]], %[[C1]], %[[C256]], %[[C1]], %[[C1]]) ({
