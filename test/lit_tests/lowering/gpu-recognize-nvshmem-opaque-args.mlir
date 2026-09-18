// RUN: not enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" 2>&1 | FileCheck %s

// The args array is handed to a function this pass does not model, which may
// write any slot. Reconstructing the launch from the stores it can see would
// mean assuming that call writes nothing, so the pass reports instead.

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @nvshmemx_collective_launch(!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
  llvm.func @fill_args(!llvm.ptr, !llvm.ptr)

  llvm.func @"reactant$_Z18__device_stub__fooPi"(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
    llvm.return
  }

  llvm.func @host_stub(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %dev = llvm.mlir.addressof @"reactant$_Z18__device_stub__fooPi" : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%dev, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c0_i64, %null, %arg0) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @main(%arg0: !llvm.ptr) -> i32 {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %grid_xy = llvm.mlir.constant(4294967297 : i64) : i64
    %block_xy = llvm.mlir.constant(4294967424 : i64) : i64
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %stub = llvm.mlir.addressof @host_stub : !llvm.ptr
    %args = llvm.alloca %c1_i32 x !llvm.array<1 x ptr> : (i32) -> !llvm.ptr
    llvm.call @fill_args(%args, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    %r = llvm.call @nvshmemx_collective_launch(%stub, %grid_xy, %c1_i32, %block_xy, %c1_i32, %args, %c0_i64, %null) : (!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// CHECK: error: nvshmemx_collective_launch: could not statically resolve args array
