// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=rocm})" | FileCheck %s --check-prefix=ROCM

module attributes {gpu.container_module} {
  llvm.mlir.global external @device_var(dense<0.000000e+00> : tensor<4xf32>) {addr_space = 1 : i32} : !llvm.array<4 x f32>

  llvm.func @launch(%arg0: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1_i64 = arith.constant 1 : i64
    %stream = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
    gpu.launch_func <%stream : !llvm.ptr> @test_module::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : !llvm.ptr)
    llvm.return
  }

  gpu.module @test_module {
    llvm.mlir.global external @device_var(dense<0.000000e+00> : tensor<4xf32>) {addr_space = 1 : i32} : !llvm.array<4 x f32>

    gpu.func @test_kernel(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
}

// A device global is registered with the runtime the rest of the module is
// registered with, not with the mlir gpu runtime.

// CHECK: llvm.func @__cudaRegisterVar(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i64, i32, i32)
// CHECK: llvm.call @__cudaRegisterVar(
// CHECK-NOT: __mgpurtRegisterVar

// ROCM: llvm.func @__hipRegisterVar(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i64, i32, i32)
// ROCM: llvm.call @__hipRegisterVar(
// ROCM-NOT: __mgpurtRegisterVar
