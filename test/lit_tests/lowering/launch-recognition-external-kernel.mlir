// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition{backend=cuda})" | FileCheck %s

// A kernel referenced only through runtime queries whose device body lives in
// another translation unit is a bare declaration; it cannot be raised into a
// gpu.func (which requires a body), so recognition must leave it and its
// users untouched.

module {
  llvm.func @cudaFuncGetAttributes(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @"reactant$_Z19__device_stub__kernILi3EEvPd"(!llvm.ptr) attributes {target_cpu = "x86-64"}
  llvm.func @query() {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %attrs = llvm.alloca %c1 x !llvm.array<56 x i8> : (i32) -> !llvm.ptr
    %fn = llvm.mlir.addressof @"reactant$_Z19__device_stub__kernILi3EEvPd" : !llvm.ptr
    %r = llvm.call @cudaFuncGetAttributes(%attrs, %fn) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
}

// CHECK-NOT: gpu.module
// CHECK: llvm.func @reactant$_Z19__device_stub__kernILi3EEvPd(!llvm.ptr)
// CHECK: %[[FN:.+]] = llvm.mlir.addressof @reactant$_Z19__device_stub__kernILi3EEvPd
// CHECK: llvm.call @cudaFuncGetAttributes(%{{.+}}, %[[FN]])
