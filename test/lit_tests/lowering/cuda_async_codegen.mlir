// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  llvm.func weak_odr local_unnamed_addr @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii(%arg0: i64, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32) -> !llvm.ptr {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %1 = llvm.alloca %0 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.8.1", packed (struct<".2", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::cuda_cub::detail::cuda_error_category.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::error_category.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
    %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.189.1", packed (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::device_ptr.56.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::pointer.57.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::iterator_adaptor.58.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %8 = "enzymexla.gpu_error"() ({
      %9 = arith.cmpi sge, %arg0, %c1_i64 : i64
      scf.if %9 {
        gpu.launch_func <%2 : !llvm.ptr> @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_29::@_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel blocks in (%c1, %c8, %c1) threads in (%c32, %c1, %c1)  args(%1 : !llvm.ptr, %3 : !llvm.ptr, %6 : index, %4 : index, %5 : index)
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : () -> index
    llvm.unreachable
  }
  gpu.module @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_29 [#nvvm.target] {
    gpu.func @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: index, %arg4: index, %arg5: index) kernel attributes {known_block_size = array<i32: 4, 256, 1>} {
      gpu.return
    }
  }
}


// CHECK: llvm.func weak_odr local_unnamed_addr @_ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii(%arg0: i64, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32) -> !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.mlir.addressof @__polygeist__ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_29__ZN9AllocatorI8Vector_dIiEE6resizeEPS1_ii_kernel_device_stub : !llvm.ptr
// CHECK-NEXT:    %2 = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %3 = llvm.mlir.constant(32 : i64) : i64
// CHECK-NEXT:    %4 = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT:    %5 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %6 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %7 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %8 = llvm.alloca %7 x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %9 = llvm.alloca %5 x i64 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %10 = llvm.alloca %5 x i64 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %11 = llvm.alloca %5 x i64 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %12 = llvm.alloca %5 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %13 = llvm.alloca %5 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %14 = llvm.alloca %4 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %15 = llvm.alloca %7 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.8.1", packed (struct<".2", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::cuda_cub::detail::cuda_error_category.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::system::error_category.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %16 = llvm.inttoptr %7 : i64 to !llvm.ptr
// CHECK-NEXT:    %17 = llvm.alloca %5 x !llvm.struct<"struct.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::cuda_cub::__uninitialized_fill::functor.189.1", packed (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::device_ptr.56.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::pointer.57.1", (struct<"class.thrust::THRUST_200802_SM___CUDA_ARCH_LIST___NS::iterator_adaptor.58.1", (ptr)>)>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %18 = llvm.sext %arg3 : i32 to i64
// CHECK-NEXT:    %19 = llvm.sext %arg3 : i32 to i64
// CHECK-NEXT:    %20 = llvm.sext %arg3 : i32 to i64
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %21 = llvm.icmp "sge" %arg0, %7 : i64
// CHECK-NEXT:    llvm.cond_br %21, ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    llvm.store %15, %13 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %13, %14 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %17, %12 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %22 = llvm.getelementptr %14[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %12, %22 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %20, %11 : i64, !llvm.ptr
// CHECK-NEXT:    %23 = llvm.getelementptr %14[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %11, %23 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %18, %10 : i64, !llvm.ptr
// CHECK-NEXT:    %24 = llvm.getelementptr %14[3] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %10, %24 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %19, %9 : i64, !llvm.ptr
// CHECK-NEXT:    %25 = llvm.getelementptr %14[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %9, %25 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %26 = llvm.trunc %7 : i64 to i32
// CHECK-NEXT:    %27 = llvm.trunc %2 : i64 to i32
// CHECK-NEXT:    %28 = llvm.trunc %7 : i64 to i32
// CHECK-NEXT:    %29 = llvm.zext %26 : i32 to i64
// CHECK-NEXT:    %30 = llvm.zext %27 : i32 to i64
// CHECK-NEXT:    %31 = llvm.shl %30, %3 : i64
// CHECK-NEXT:    %32 = llvm.or %29, %31 : i64
// CHECK-NEXT:    %33 = llvm.trunc %3 : i64 to i32
// CHECK-NEXT:    %34 = llvm.trunc %7 : i64 to i32
// CHECK-NEXT:    %35 = llvm.trunc %7 : i64 to i32
// CHECK-NEXT:    %36 = llvm.zext %33 : i32 to i64
// CHECK-NEXT:    %37 = llvm.zext %34 : i32 to i64
// CHECK-NEXT:    %38 = llvm.shl %37, %3 : i64
// CHECK-NEXT:    %39 = llvm.or %36, %38 : i64
// CHECK-NEXT:    %40 = llvm.call @cudaLaunchKernel(%1, %32, %28, %39, %35, %14, %0, %16) : (!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.store %40, %8 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb3
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %41 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.br ^bb4(%41 : i32)
// CHECK-NEXT:  ^bb4(%42: i32):  // pred: ^bb3
// CHECK-NEXT:    llvm.store %6, %8 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.unreachable
// CHECK-NEXT:  }
