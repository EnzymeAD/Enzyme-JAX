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

// CHECK:  llvm.func @foo
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(32 : i64) : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(11 : i32) : i32
// CHECK-NEXT:    %3 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %4 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %5 = llvm.mlir.addressof @__polygeist__Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel__Z8hpl_mainii10KernelTypeRSt6vectorIdSaIdEERS0_IiSaIiEE_kernel_device_stub : !llvm.ptr
// CHECK-NEXT:    %6 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.cond_br %arg0, ^bb2, ^bb3(%6 : i32)
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %7 = llvm.alloca %3 x !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)> : (i32) -> !llvm.ptr
// CHECK-NEXT:    %8 = llvm.alloca %2 x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT:    %9 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg7, %9 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %9, %8 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %10 = llvm.getelementptr %7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg8, %10 : i32, !llvm.ptr
// CHECK-NEXT:    %11 = llvm.getelementptr %8[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %10, %11 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %12 = llvm.getelementptr %7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg9, %12 : i32, !llvm.ptr
// CHECK-NEXT:    %13 = llvm.getelementptr %8[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %12, %13 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %14 = llvm.getelementptr %7[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg10, %14 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %15 = llvm.getelementptr %8[3] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %14, %15 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %16 = llvm.getelementptr %7[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg11, %16 : i32, !llvm.ptr
// CHECK-NEXT:    %17 = llvm.getelementptr %8[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %16, %17 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %18 = llvm.getelementptr %7[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg12, %18 : i32, !llvm.ptr
// CHECK-NEXT:    %19 = llvm.getelementptr %8[5] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %18, %19 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %20 = llvm.getelementptr %7[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg13, %20 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %21 = llvm.getelementptr %8[6] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %20, %21 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %22 = llvm.getelementptr %7[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg14, %22 : f64, !llvm.ptr
// CHECK-NEXT:    %23 = llvm.getelementptr %8[7] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %22, %23 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %24 = llvm.getelementptr %7[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg15, %24 : i32, !llvm.ptr
// CHECK-NEXT:    %25 = llvm.getelementptr %8[8] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %24, %25 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %26 = llvm.getelementptr %7[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg16, %26 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %27 = llvm.getelementptr %8[9] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %26, %27 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %28 = llvm.getelementptr %7[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"", (i32, i32, i32, ptr, i32, i32, ptr, f64, i32, ptr, f64)>
// CHECK-NEXT:    llvm.store %arg17, %28 : f64, !llvm.ptr
// CHECK-NEXT:    %29 = llvm.getelementptr %8[10] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %28, %29 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %30 = llvm.trunc %arg1 : i64 to i32
// CHECK-NEXT:    %31 = llvm.trunc %arg2 : i64 to i32
// CHECK-NEXT:    %32 = llvm.trunc %arg3 : i64 to i32
// CHECK-NEXT:    %33 = llvm.zext %30 : i32 to i64
// CHECK-NEXT:    %34 = llvm.zext %31 : i32 to i64
// CHECK-NEXT:    %35 = llvm.shl %34, %1 : i64
// CHECK-NEXT:    %36 = llvm.or %33, %35 : i64
// CHECK-NEXT:    %37 = llvm.trunc %arg4 : i64 to i32
// CHECK-NEXT:    %38 = llvm.trunc %arg5 : i64 to i32
// CHECK-NEXT:    %39 = llvm.trunc %arg6 : i64 to i32
// CHECK-NEXT:    %40 = llvm.zext %37 : i32 to i64
// CHECK-NEXT:    %41 = llvm.zext %38 : i32 to i64
// CHECK-NEXT:    %42 = llvm.shl %41, %1 : i64
// CHECK-NEXT:    %43 = llvm.or %40, %42 : i64
// CHECK-NEXT:    %44 = llvm.call @cudaLaunchKernel(%5, %36, %32, %43, %39, %8, %0, %4) : (!llvm.ptr, i64, i32, i64, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.br ^bb3(%44 : i32)
// CHECK-NEXT:  ^bb3(%45: i32):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %46 = llvm.sext %45 : i32 to i64
// CHECK-NEXT:    llvm.br ^bb4
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    llvm.return %46 : i64
// CHECK-NEXT:  }
