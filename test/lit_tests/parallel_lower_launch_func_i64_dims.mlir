// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower{wrapParallelOps=true})" | FileCheck %s

// gpu.launch_func takes i32 or i64 dimensions (launch recognition passes the
// sign-extended dim3 fields of a kernel whose address escapes, as MFEM's dfem
// kernels do); the gpu.launch and enzymexla.gpu_wrapper it is lowered to take
// index. The kernel, in the GPU module where no func.call can name it, is
// inlined into the launch, and goes once nothing else uses it.

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

// CHECK:       gpu.module @__mlir_gpu_module {
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @host(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: i32) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = arith.constant 256 : index
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = llvm.sext %[[V1]] : i32 to i64
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = arith.index_cast %[[V6]] : i64 to index
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = "enzymexla.gpu_wrapper"(%[[V7]], %[[V4]], %[[V4]], %[[V2]], %[[V4]], %[[V4]]) ({
// CHECK-NEXT: scf.parallel (%[[V9:[^ ,:)]+]], %[[V10:[^ ,:)]+]], %[[V11:[^ ,:)]+]]) = (%[[V5]], %[[V5]], %[[V5]]) to (%[[V7]], %[[V4]], %[[V4]]) step (%[[V4]], %[[V4]], %[[V4]]) {
// CHECK-NEXT: scf.parallel (%[[V12:[^ ,:)]+]], %[[V13:[^ ,:)]+]], %[[V14:[^ ,:)]+]]) = (%[[V5]], %[[V5]], %[[V5]]) to (%[[V2]], %[[V4]], %[[V4]]) step (%[[V4]], %[[V4]], %[[V4]]) {
// CHECK-NEXT: memref.alloca_scope  {
// CHECK-NEXT: scf.execute_region {
// CHECK-NEXT: llvm.store %[[V3]], %[[V0]] : f64, !llvm.ptr
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: scf.reduce
// CHECK-NEXT: }
// CHECK-NEXT: scf.reduce
// CHECK-NEXT: }
// CHECK-NEXT: "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT: }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT: return
// CHECK-NEXT: }
