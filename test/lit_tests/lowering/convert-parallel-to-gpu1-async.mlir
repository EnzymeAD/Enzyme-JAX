// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

module attributes { gpu.container_module } {
  llvm.func weak_odr local_unnamed_addr @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii(%sz: i64, %arg0: !llvm.ptr, %arg1: i32, %arg2: i32) -> (!llvm.ptr)  {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %1 = llvm.alloca %0 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.8.1", packed (struct<".2", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::cuda_cub::detail::cuda_error_category.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::error_category.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
    %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.189.1", packed (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::device_ptr.56.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::pointer.57.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::iterator_adaptor.58.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg2 : i32 to index
    %7 = "enzymexla.stream2token"(%2) : (!llvm.ptr) -> !async.token
    %token = async.execute [%7] {
      %8 = "enzymexla.gpu_error"() ({
        %156 = arith.cmpi sge, %sz, %c1_i64 : i64
          scf.if %156 {
            gpu.launch_func  @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_29::@_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel blocks in (%c1, %c8, %c1) threads in (%c32, %c1, %c1)  args(%1 : !llvm.ptr, %3 : !llvm.ptr, %6 : index, %4 : index, %5 : index)
	  }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      async.yield
    }
    llvm.unreachable
  }
  gpu.module @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_29 [#nvvm.target] {
    gpu.func @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: index, %arg4: index, %arg5: index) kernel attributes {known_block_size = array<i32: 4, 256, 1>} {
      gpu.return
    }
  }
}

// CHECK:  llvm.func weak_odr local_unnamed_addr @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii(%arg0: i64, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32) -> !llvm.ptr {
// CHECK-NEXT:    %c32 = arith.constant 32 : index
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %1 = llvm.alloca %0 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.8.1", packed (struct<".2", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::cuda_cub::detail::cuda_error_category.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::error_category.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %2 = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
// CHECK-NEXT:    %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.189.1", packed (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::device_ptr.56.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::pointer.57.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::iterator_adaptor.58.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %4 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:    %5 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:    %6 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:    %7 = "enzymexla.stream2token"(%2) : (!llvm.ptr) -> !gpu.async.token
// CHECK-NEXT:    %8 = "enzymexla.gpu_error"() ({
// CHECK-NEXT:      %9 = arith.cmpi sge, %arg0, %c1_i64 : i64
// CHECK-NEXT:      scf.if %9 {
// CHECK-NEXT:        gpu.launch_func [%7] @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_29::@_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel blocks in (%c1, %c8, %c1) threads in (%c32, %c1, %c1)  args(%1 : !llvm.ptr, %3 : !llvm.ptr, %6 : index, %4 : index, %5 : index)
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : () -> index
// CHECK-NEXT:    llvm.unreachable
// CHECK-NEXT:  }

