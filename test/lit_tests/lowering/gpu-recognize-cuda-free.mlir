// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-enzymexla-to-llvm{backend=cuda})" | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-enzymexla-to-llvm{backend=cpu})" | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-enzymexla-to-llvm{backend=xla-tpu})" | FileCheck %s --check-prefix=XLA

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  llvm.func @test_cuda_free_cast(%arg: !llvm.ptr<1>) {
    %0 = "enzymexla.pointer2memref"(%arg) : (!llvm.ptr<1>) -> memref<?xi8, 1>
    gpu.dealloc %0 : memref<?xi8, 1>
    llvm.return
  }
}

// CUDA-LABEL: llvm.func @test_cuda_free_cast(%arg0: !llvm.ptr<1>)
// CUDA: %[[CAST:.*]] = llvm.addrspacecast %arg0 : !llvm.ptr<1> to !llvm.ptr
// CUDA: llvm.call @cudaFree(%[[CAST]]) : (!llvm.ptr) -> i32

// CPU-LABEL: llvm.func @test_cuda_free_cast(%arg0: !llvm.ptr<1>)
// CPU: %[[CAST:.*]] = llvm.addrspacecast %arg0 : !llvm.ptr<1> to !llvm.ptr
// CPU: llvm.call @free(%[[CAST]]) : (!llvm.ptr) -> ()

// XLA-LABEL: llvm.func @test_cuda_free_cast(%arg0: !llvm.ptr<1>)
// XLA: %[[CAST:.*]] = llvm.addrspacecast %arg0 : !llvm.ptr<1> to !llvm.ptr
// XLA: llvm.call @reactantXLAFree(


