// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func local_unnamed_addr @main(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(512 : i64) : i64
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.addressof @reactant$_Z18__device_stub__fooPi : !llvm.ptr
    %4 = llvm.mlir.constant(128 : i32) : i32
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%3, %0, %0, %0, %4, %0, %0, %5, %6, %arg0) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func internal @reactant$_Z18__device_stub__fooPi(%arg0: !llvm.ptr<1>) attributes {dso_local, sym_visibility = "private", target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88", "+sm_120"]>} {
    llvm.return
  }
  llvm.func local_unnamed_addr @__mlir_cuda_caller_phase3(...) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func local_unnamed_addr @main(%arg0: !llvm.ptr)
// CHECK: %[[CAST:.*]] = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<1>
// CHECK: gpu.launch_func  @__mlir_gpu_module::@reactant$_Z18__device_stub__fooPi blocks in ({{.*}}) threads in ({{.*}}) : i64 dynamic_shared_memory_size {{.*}} args(%[[CAST]] : !llvm.ptr<1>)
