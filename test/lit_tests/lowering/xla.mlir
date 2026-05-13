// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s

module {
  llvm.mlir.global private unnamed_addr constant @".str"("%f %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.400000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1xf64>
    %alloca_2 = memref.alloca() : memref<1xf64>
    %alloca_3 = memref.alloca() : memref<1xf64>
    %alloca_4 = memref.alloca() : memref<1xf64>
    %memref = gpu.alloc  () : memref<1xf64, 1>
    %memref_5 = gpu.alloc  () : memref<1xf64, 1>
    %memref_6 = gpu.alloc  () : memref<1xf64, 1>
    %memref_7 = gpu.alloc  () : memref<1xf64, 1>
    memref.store %cst, %alloca[%c0] : memref<1xf64>
    memref.store %cst_0, %alloca_2[%c0] : memref<1xf64>
    memref.store %cst_1, %alloca_3[%c0] : memref<1xf64>
    memref.store %cst_1, %alloca_4[%c0] : memref<1xf64>
    enzymexla.memcpy  %memref, %alloca, %c8 : memref<1xf64, 1>, memref<1xf64>
    enzymexla.memcpy  %memref_5, %alloca_2, %c8 : memref<1xf64, 1>, memref<1xf64>
    enzymexla.memcpy  %memref_6, %alloca_3, %c8 : memref<1xf64, 1>, memref<1xf64>
    enzymexla.memcpy  %memref_7, %alloca_4, %c8 : memref<1xf64, 1>, memref<1xf64>
    enzymexla.xla_wrapper @raised (%memref, %memref_6) : (memref<1xf64, 1>, memref<1xf64, 1>) -> ()
    %1 = llvm.call @cudaDeviceSynchronize() : () -> i32
    enzymexla.memcpy  %alloca, %memref, %c8 : memref<1xf64>, memref<1xf64, 1>
    enzymexla.memcpy  %alloca_2, %memref_5, %c8 : memref<1xf64>, memref<1xf64, 1>
    enzymexla.memcpy  %alloca_3, %memref_6, %c8 : memref<1xf64>, memref<1xf64, 1>
    enzymexla.memcpy  %alloca_4, %memref_7, %c8 : memref<1xf64>, memref<1xf64, 1>
    %2 = memref.load %alloca[%c0] : memref<1xf64>
    %3 = memref.load %alloca_3[%c0] : memref<1xf64>
    %4 = llvm.call @printf(%0, %2, %3) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    %5 = memref.load %alloca_2[%c0] : memref<1xf64>
    %6 = memref.load %alloca_4[%c0] : memref<1xf64>
    %7 = llvm.call @printf(%0, %5, %6) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    llvm.return %c0_i32 : i32
  }
  llvm.func local_unnamed_addr @cudaDeviceSynchronize() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  func.func private @raised(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.reshape %arg0 : (tensor<1xf64>) -> tensor<f64>
    %1 = stablehlo.reshape %arg1 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.add %0, %1 : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %4 = stablehlo.dynamic_update_slice %arg1, %3, %c : (tensor<1xf64>, tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
    return %arg0, %4 : tensor<1xf64>, tensor<1xf64>
  }
}

// CHECK:  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %1 = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %2 = llvm.mlir.addressof @xlamod$raised : !llvm.ptr
// CHECK-NEXT:    %3 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %4 = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// CHECK-NEXT:    %5 = llvm.mlir.constant(12 : i64) : i64
// CHECK-NEXT:    %6 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %7 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %8 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:    %9 = llvm.mlir.constant(1.000000e+00 : f64) : f64
// CHECK-NEXT:    %10 = llvm.mlir.constant(0.000000e+00 : f64) : f64
// CHECK-NEXT:    %11 = llvm.mlir.constant(1.400000e+00 : f64) : f64
// CHECK-NEXT:    %12 = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:    %13 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %14 = llvm.alloca %13 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %15 = llvm.alloca %13 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %16 = llvm.alloca %13 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %17 = llvm.alloca %13 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %18 = llvm.alloca %13 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %19 = llvm.alloca %6 x f64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %20 = llvm.alloca %6 x f64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %21 = llvm.alloca %6 x f64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %22 = llvm.alloca %6 x f64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %23 = llvm.getelementptr %18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
// CHECK-NEXT:    llvm.store %13, %23 : i64, !llvm.ptr
// CHECK-NEXT:    %24 = llvm.call @reactantXLAMalloc(%4, %5, %13, %18) : (!llvm.ptr, i64, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %25 = llvm.addrspacecast %24 : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %26 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
// CHECK-NEXT:    llvm.store %13, %26 : i64, !llvm.ptr
// CHECK-NEXT:    %27 = llvm.call @reactantXLAMalloc(%4, %5, %13, %17) : (!llvm.ptr, i64, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %28 = llvm.getelementptr %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
// CHECK-NEXT:    llvm.store %13, %28 : i64, !llvm.ptr
// CHECK-NEXT:    %29 = llvm.call @reactantXLAMalloc(%4, %5, %13, %16) : (!llvm.ptr, i64, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %30 = llvm.addrspacecast %29 : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %31 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
// CHECK-NEXT:    llvm.store %13, %31 : i64, !llvm.ptr
// CHECK-NEXT:    %32 = llvm.call @reactantXLAMalloc(%4, %5, %13, %15) : (!llvm.ptr, i64, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %11, %19 : f64, !llvm.ptr
// CHECK-NEXT:    llvm.store %10, %20 : f64, !llvm.ptr
// CHECK-NEXT:    llvm.store %9, %21 : f64, !llvm.ptr
// CHECK-NEXT:    llvm.store %9, %22 : f64, !llvm.ptr
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %24, %19, %12, %3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %27, %20, %12, %3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %29, %21, %12, %3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %32, %22, %12, %3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    %33 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<574 x i8>
// CHECK-NEXT:    %34 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
// CHECK-NEXT:    llvm.store %25, %34 : !llvm.ptr<1>, !llvm.ptr
// CHECK-NEXT:    %35 = llvm.getelementptr %14[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
// CHECK-NEXT:    llvm.store %30, %35 : !llvm.ptr<1>, !llvm.ptr
// CHECK-NEXT:    llvm.call @reactantXLAExec(%4, %33, %1, %14) vararg(!llvm.func<void (ptr, ptr, i64, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
// CHECK-NEXT:    %36 = llvm.mlir.zero : i32
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %19, %24, %12, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %20, %27, %12, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %21, %29, %12, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%4, %22, %32, %12, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK-NEXT:    %37 = llvm.load %19 : !llvm.ptr -> f64
// CHECK-NEXT:    %38 = llvm.load %21 : !llvm.ptr -> f64
// CHECK-NEXT:    %39 = llvm.call @printf(%8, %37, %38) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
// CHECK-NEXT:    %40 = llvm.load %20 : !llvm.ptr -> f64
// CHECK-NEXT:    %41 = llvm.load %22 : !llvm.ptr -> f64
// CHECK-NEXT:    %42 = llvm.call @printf(%8, %40, %41) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
// CHECK-NEXT:    llvm.return %7 : i32
// CHECK-NEXT:  }

// CHECK:  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
// CHECK-NEXT:  llvm.func linkonce @__reactant_xla_init() {
// CHECK-NEXT:    %0 = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// CHECK-NEXT:    %1 = llvm.mlir.addressof @xlabackend : !llvm.ptr
// CHECK-NEXT:    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
// CHECK-NEXT:    llvm.call @reactantXLAInit(%0, %2) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK:  llvm.func linkonce @__reactant_xla_deinit() {
// CHECK-NEXT:    %0 = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// CHECK-NEXT:    llvm.call @reactantXLADeInit(%0) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK:  llvm.mlir.global_ctors ctors = [@__reactant_xla_init], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK:  llvm.mlir.global_dtors dtors = [@__reactant_xla_deinit], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK:  llvm.mlir.global linkonce @__reactant_xla_data() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
