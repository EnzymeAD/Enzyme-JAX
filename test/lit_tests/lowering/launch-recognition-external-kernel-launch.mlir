// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition{backend=cuda})" | FileCheck %s

// The host launches a kernel the device compilation never instantiated (a bare
// declaration). There is nothing to raise; the launch fails when it runs, as it
// does natively, instead of leaving the launch placeholder undefined at link
// time.

module {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @"reactant$_Z19__device_stub__kernILi24EEvPd"(!llvm.ptr)
  llvm.func @caller(%x: !llvm.ptr) {
    %f = llvm.mlir.addressof @"reactant$_Z19__device_stub__kernILi24EEvPd" : !llvm.ptr
    %dim = llvm.mlir.constant(1 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%f, %dim, %dim, %dim, %dim, %dim, %dim, %shmem, %stream, %x) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}

// CHECK-NOT: gpu.module
// CHECK: llvm.mlir.global internal constant @__reactant_no_device_code.reactant$_Z19__device_stub__kernILi24EEvPd("launch of reactant$_Z19__device_stub__kernILi24EEvPd, which has no device code\00") {addr_space = 0 : i32}
// CHECK-NEXT: llvm.func @abort()
// CHECK-NEXT: llvm.func @puts(!llvm.ptr) -> i32
// CHECK-NEXT: llvm.func @__mlir_cuda_caller_phase3(...)
// CHECK-NEXT: llvm.func @reactant$_Z19__device_stub__kernILi24EEvPd(!llvm.ptr)
// CHECK-NEXT: llvm.func @caller(%[[X:.+]]: !llvm.ptr) {
// CHECK-NEXT:   %[[F:.+]] = llvm.mlir.addressof @reactant$_Z19__device_stub__kernILi24EEvPd : !llvm.ptr
// CHECK-NEXT:   %[[DIM:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:   %[[SHMEM:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:   %[[STREAM:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:   %[[MSG:.+]] = llvm.mlir.addressof @__reactant_no_device_code.reactant$_Z19__device_stub__kernILi24EEvPd : !llvm.ptr
// CHECK-NEXT:   %{{.+}} = llvm.call @puts(%[[MSG]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:   llvm.call @abort() : () -> ()
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }
