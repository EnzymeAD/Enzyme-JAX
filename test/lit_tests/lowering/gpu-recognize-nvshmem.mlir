// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

// nvshmemx_collective_launch takes the *host* stub by address; the device stub
// is one indirection further in, behind the __mlir_cuda_caller_phase3 call
// inside that host stub. The host stub has no direct callers, only an
// addressof use, which is what distinguishes it from an ordinary entry point.
//
// dim3 reaches the call as a packed {x,y} i64 plus a separate z, so the grid
// and block extents have to be unpacked before they can feed a gpu.launch.

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @nvshmemx_collective_launch(!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32

  // Device stub for the kernel.
  llvm.func @"reactant$_Z18__device_stub__fooPi"(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
    llvm.return
  }

  // Host stub: never called directly, only taken by address and handed to
  // nvshmemx_collective_launch. It forwards to the phase3 caller.
  llvm.func @host_stub(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %c128_i32 = llvm.mlir.constant(128 : i32) : i32
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %dev = llvm.mlir.addressof @"reactant$_Z18__device_stub__fooPi" : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%dev, %c1_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %c0_i64, %null, %arg0) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @main(%arg0: !llvm.ptr) -> i32 {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    // dim3 arrives as a packed {x,y} i64 plus a separate z.
    %grid_xy = llvm.mlir.constant(4294967297 : i64) : i64
    %block_xy = llvm.mlir.constant(4294967424 : i64) : i64
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %stub = llvm.mlir.addressof @host_stub : !llvm.ptr
    // void *args[1] = { &devPtr };
    %args = llvm.alloca %c1_i32 x !llvm.array<1 x ptr> : (i32) -> !llvm.ptr
    llvm.store %arg0, %args : !llvm.ptr, !llvm.ptr
    %r = llvm.call @nvshmemx_collective_launch(%stub, %grid_xy, %c1_i32, %block_xy, %c1_i32, %args, %c0_i64, %null) : (!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// CHECK-LABEL: llvm.func @main(
// The packed dim3 pair is split back into x/y and index-cast for gpu.launch.
// CHECK:         %[[GX:.+]] = arith.trunci %{{.*}} : i64 to i32
// CHECK:         %[[GYS:.+]] = arith.shrui %{{.*}}, %{{.*}} : i64
// CHECK:         %[[GY:.+]] = arith.trunci %[[GYS]] : i64 to i32
// CHECK:         arith.index_cast %[[GX]] : i32 to index
// CHECK:         arith.index_cast %[[GY]] : i32 to index
// CHECK:         gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in ({{.*}}) dynamic_shared_memory_size
// The kernel argument is read out of the void** args array.
// CHECK:           llvm.call @reactant$_Z18__device_stub__fooPi(%{{.*}}) : (!llvm.ptr) -> ()
// CHECK:           gpu.terminator
// CHECK-NOT:     nvshmemx_collective_launch
