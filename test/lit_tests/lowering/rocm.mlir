// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=rocm})" | FileCheck %s

module attributes {gpu.container_module} {
  llvm.func @test_rocm_launch(%arg0: !llvm.ptr) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %stream = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
    %token = "enzymexla.stream2token"(%stream) : (!llvm.ptr) -> !gpu.async.token
    gpu.launch_func [%token] @test_module::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : !llvm.ptr)
    llvm.return
  }

  func.func @test_rocm_alloc() {
    %alloc = gpu.alloc() : memref<256xf32, 1>
    gpu.dealloc %alloc : memref<256xf32, 1>
    return
  }

  func.func @test_rocm_memcpy(%src: memref<256xf32>, %dst: memref<256xf32, 1>) {
    %c1024 = arith.constant 1024 : index
    "enzymexla.memcpy"(%dst, %src, %c1024) : (memref<256xf32, 1>, memref<256xf32>, index) -> ()
    return
  }

  gpu.module @test_module {
    gpu.func @test_kernel(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @test_rocm_launch
// CHECK: llvm.call @hipLaunchKernel
// CHECK-NOT: cudaLaunchKernel

// CHECK-LABEL: llvm.func @test_rocm_alloc
// CHECK: llvm.call @hipMalloc
// CHECK: llvm.call @hipFree
// CHECK-NOT: cudaMalloc
// CHECK-NOT: cudaFree

// CHECK-LABEL: llvm.func @test_rocm_memcpy
// CHECK: llvm.call @hipMemcpy
// CHECK-NOT: cudaMemcpy
