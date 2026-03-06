// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=rocm})" | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, gpu.container_module, llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %memref = gpu.alloc () : memref<100xf64, 1>
    %memref_6 = gpu.alloc () : memref<100xf64, 1>
    %0 = "enzymexla.gpu_error"() ({
      gpu.launch_func @main_kernel::@main_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%memref : memref<100xf64, 1>, %memref_6 : memref<100xf64, 1>)
      "enzymexla.polygeist_yield"() : () -> ()
    }) : () -> index
    llvm.return %c0_i32 : i32
  }

  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<100xf64, 1>, %arg1: memref<100xf64, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c100 = arith.constant 100 : index
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id x
      %thread_id_x = gpu.thread_id x
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %c100 : index
      scf.if %2 {
        %3 = memref.load %arg0[%1] : memref<100xf64, 1>
        %4 = memref.load %arg1[%1] : memref<100xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg1[%1] : memref<100xf64, 1>
      }
      gpu.return
    }
  }
}

// CHECK-LABEL: @main
// CHECK: gpu.module @main_kernel [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}