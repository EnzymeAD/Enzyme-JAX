// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

// Two kernel parameters, so the void** array is indexed rather than written at
// offset zero. Slot i is found through the getelementptr, not by measuring a
// byte offset, and the slots keep their order regardless of the order the
// stores appear in.
//
// The first slot holds a pointer the kernel takes directly; the second holds a
// void* pointing at the i32 the kernel wants, so it has to be loaded.

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @nvshmemx_collective_launch(!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32

  llvm.func @"reactant$_Z18__device_stub__barPii"(%arg0: !llvm.ptr, %arg1: i32) attributes {sym_visibility = "private"} {
    llvm.return
  }

  llvm.func @host_stub(%arg0: !llvm.ptr, %arg1: i32) attributes {sym_visibility = "private"} {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %dev = llvm.mlir.addressof @"reactant$_Z18__device_stub__barPii" : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%dev, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c0_i64, %null, %arg0, %arg1) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.return
  }

  llvm.func @main(%devPtr: !llvm.ptr, %scalarPtr: !llvm.ptr) -> i32 {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %grid_xy = llvm.mlir.constant(4294967297 : i64) : i64
    %block_xy = llvm.mlir.constant(4294967424 : i64) : i64
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %stub = llvm.mlir.addressof @host_stub : !llvm.ptr
    // void *args[2]; args[1] = &scalar; args[0] = devPtr;
    %args = llvm.alloca %c1_i32 x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
    %slot1 = llvm.getelementptr %args[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    llvm.store %scalarPtr, %slot1 : !llvm.ptr, !llvm.ptr
    %slot0 = llvm.getelementptr %args[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    llvm.store %devPtr, %slot0 : !llvm.ptr, !llvm.ptr
    %r = llvm.call @nvshmemx_collective_launch(%stub, %grid_xy, %c1_i32, %block_xy, %c1_i32, %args, %c0_i64, %null) : (!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// CHECK-LABEL: llvm.func @main(
// CHECK-SAME:      %[[DEV:.+]]: !llvm.ptr, %[[SCALAR:.+]]: !llvm.ptr
// The i32 parameter is read out of the slot; the pointer parameter is not.
// CHECK:         %[[MEM:.+]] = "enzymexla.pointer2memref"(%[[SCALAR]])
// CHECK:         %[[VAL:.+]] = memref.load %[[MEM]]
// CHECK:         gpu.launch blocks
// CHECK:           llvm.call @reactant$_Z18__device_stub__barPii(%[[DEV]], %[[VAL]])
// CHECK:           gpu.terminator
// CHECK-NOT:     nvshmemx_collective_launch
