// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, gpu.container_module, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @foo(%315: i1, %274 : index, %275 : index, %276 : index, %277 : index, %278 : index, %279 : index, %243 : i32, %237 : i32, %252 : i32, %249 : !llvm.ptr, %240 : i32, %258 : i32, %255 : !llvm.ptr, %246 : f64, %267 : i32, %264 : !llvm.ptr, %261 : f64) -> index {
    %c1 = arith.constant 1 : index
    %280 = "enzymexla.gpu_error"() ({
      scf.if %315 {
        gpu.launch_func  @_Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel::@_Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel blocks in (%274, %275, %276) threads in (%277, %278, %279)  args(%243 : i32, %237 : i32, %252 : i32, %249 : !llvm.ptr, %240 : i32, %258 : i32, %255 : !llvm.ptr, %246 : f64, %267 : i32, %264 : !llvm.ptr, %261 : f64)
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : () -> index
    func.return %280 : index
  }
  gpu.module @_Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel [#nvvm.target] {
    gpu.func @_Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !llvm.ptr, %arg4: i32, %arg5: i32, %arg6: !llvm.ptr, %arg7: f64, %arg8: i32, %arg9: !llvm.ptr, %arg10: f64) kernel {
      gpu.return
    }
  }
}

// CHECK:  llvm.func @foo(%arg0: i1, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: !llvm.ptr, %arg11: i32, %arg12: i32, %arg13: !llvm.ptr, %arg14: f64, %arg15: i32, %arg16: !llvm.ptr, %arg17: f64) -> i64 {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(32 : i64) : i64
// CHECK-NEXT:    %2 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %3 = llvm.mlir.addressof @__polygeist__Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel__Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel_device_stub : !llvm.ptr
// CHECK-NEXT:    %4 = llvm.mlir.constant(11 : i32) : i32
// CHECK-NEXT:    %5 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %6 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %7 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %8 = llvm.alloca %7 x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %9 = llvm.alloca %5 x f64 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %10 = llvm.alloca %5 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %11 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %12 = llvm.alloca %5 x f64 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %13 = llvm.alloca %5 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %14 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %15 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %16 = llvm.alloca %5 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %17 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %18 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %19 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
// CHECK-NEXT:    %20 = llvm.alloca %4 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.cond_br %arg0, ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    llvm.store %arg7, %19 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %19, %20 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg8, %18 : i32, !llvm.ptr
// CHECK-NEXT:    %21 = llvm.getelementptr %20[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %18, %21 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg9, %17 : i32, !llvm.ptr
// CHECK-NEXT:    %22 = llvm.getelementptr %20[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %17, %22 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg10, %16 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %23 = llvm.getelementptr %20[3] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %16, %23 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg11, %15 : i32, !llvm.ptr
// CHECK-NEXT:    %24 = llvm.getelementptr %20[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %15, %24 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg12, %14 : i32, !llvm.ptr
// CHECK-NEXT:    %25 = llvm.getelementptr %20[5] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %14, %25 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg13, %13 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %26 = llvm.getelementptr %20[6] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %13, %26 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg14, %12 : f64, !llvm.ptr
// CHECK-NEXT:    %27 = llvm.getelementptr %20[7] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %12, %27 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg15, %11 : i32, !llvm.ptr
// CHECK-NEXT:    %28 = llvm.getelementptr %20[8] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %11, %28 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg16, %10 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %29 = llvm.getelementptr %20[9] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %10, %29 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %arg17, %9 : f64, !llvm.ptr
// CHECK-NEXT:    %30 = llvm.getelementptr %20[10] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %9, %30 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %31 = llvm.trunc %arg1 : i64 to i32
// CHECK-NEXT:    %32 = llvm.trunc %arg2 : i64 to i32
// CHECK-NEXT:    %33 = llvm.trunc %arg3 : i64 to i32
// CHECK-NEXT:    %34 = llvm.zext %31 : i32 to i64
// CHECK-NEXT:    %35 = llvm.zext %32 : i32 to i64
// CHECK-NEXT:    %36 = llvm.shl %35, %1 : i64
// CHECK-NEXT:    %37 = llvm.or %34, %36 : i64
// CHECK-NEXT:    %38 = llvm.trunc %arg4 : i64 to i32
// CHECK-NEXT:    %39 = llvm.trunc %arg5 : i64 to i32
// CHECK-NEXT:    %40 = llvm.trunc %arg6 : i64 to i32
// CHECK-NEXT:    %41 = llvm.zext %38 : i32 to i64
// CHECK-NEXT:    %42 = llvm.zext %39 : i32 to i64
// CHECK-NEXT:    %43 = llvm.shl %42, %1 : i64
// CHECK-NEXT:    %44 = llvm.or %41, %43 : i64
// CHECK-NEXT:    %45 = llvm.call @cudaLaunchKernel(%3, %37, %33, %44, %40, %20, %0, %2) : (!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.store %45, %8 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb3
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %46 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.br ^bb4(%46 : i32)
// CHECK-NEXT:  ^bb4(%47: i32):  // pred: ^bb3
// CHECK-NEXT:    llvm.store %6, %8 : i32, !llvm.ptr
// CHECK-NEXT:    %48 = llvm.sext %47 : i32 to i64
// CHECK-NEXT:    llvm.return %48 : i64
// CHECK-NEXT:  }
