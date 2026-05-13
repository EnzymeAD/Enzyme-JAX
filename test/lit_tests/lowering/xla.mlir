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
// CHECK-NEXT:    %[[C2_I32:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[C2_I64:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[RAISED:.+]] = llvm.mlir.addressof @xlamod$raised : !llvm.ptr
// CHECK-NEXT:    %[[C1_I32:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[XDATA:.+]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// CHECK-NEXT:    %[[C12:.+]] = llvm.mlir.constant(12 : i64) : i64
// CHECK-NEXT:    %[[C1_IDX:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[C0_I32:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[STR:.+]] = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:    %[[CF1:.+]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
// CHECK-NEXT:    %[[CF0:.+]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
// CHECK-NEXT:    %[[CF1_4:.+]] = llvm.mlir.constant(1.400000e+00 : f64) : f64
// CHECK-NEXT:    %[[C8_IDX:.+]] = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:    %[[C1_I64:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[ALLOCA_EXEC:.+]] = llvm.alloca %[[C1_I64]] x !llvm.array<2 x i64>
// CHECK-NEXT:    %[[ALLOCA1:.+]] = llvm.alloca %[[C1_IDX]] x f64
// CHECK-NEXT:    %[[ALLOCA2:.+]] = llvm.alloca %[[C1_IDX]] x f64
// CHECK-NEXT:    %[[ALLOCA3:.+]] = llvm.alloca %[[C1_IDX]] x f64
// CHECK-NEXT:    %[[ALLOCA4:.+]] = llvm.alloca %[[C1_IDX]] x f64
// CHECK-NEXT:    %[[ALLOCA_MALLOC:.+]] = llvm.alloca %[[C1_I64]] x !llvm.array<1 x i64>
// CHECK-NEXT:    %[[GEP:.+]] = llvm.getelementptr %[[ALLOCA_MALLOC]][0, 0]
// CHECK-NEXT:    llvm.store %[[C1_I64]], %[[GEP]]
// CHECK-NEXT:    %[[MALLOC1:.+]] = llvm.call @reactantXLAMalloc(%[[XDATA]], %[[C12]], %[[C1_I64]], %[[ALLOCA_MALLOC]])
// CHECK-NEXT:    %[[CAST1:.+]] = llvm.addrspacecast %[[MALLOC1]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[ALLOCA_MALLOC2:.+]] = llvm.alloca %[[C1_I64]] x !llvm.array<1 x i64>
// CHECK-NEXT:    %[[GEP2:.+]] = llvm.getelementptr %[[ALLOCA_MALLOC2]][0, 0]
// CHECK-NEXT:    llvm.store %[[C1_I64]], %[[GEP2]]
// CHECK-NEXT:    %[[MALLOC2:.+]] = llvm.call @reactantXLAMalloc(%[[XDATA]], %[[C12]], %[[C1_I64]], %[[ALLOCA_MALLOC2]])
// CHECK-NEXT:    %[[ALLOCA_MALLOC3:.+]] = llvm.alloca %[[C1_I64]] x !llvm.array<1 x i64>
// CHECK-NEXT:    %[[GEP3:.+]] = llvm.getelementptr %[[ALLOCA_MALLOC3]][0, 0]
// CHECK-NEXT:    llvm.store %[[C1_I64]], %[[GEP3]]
// CHECK-NEXT:    %[[MALLOC3:.+]] = llvm.call @reactantXLAMalloc(%[[XDATA]], %[[C12]], %[[C1_I64]], %[[ALLOCA_MALLOC3]])
// CHECK-NEXT:    %[[CAST3:.+]] = llvm.addrspacecast %[[MALLOC3]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[ALLOCA_MALLOC4:.+]] = llvm.alloca %[[C1_I64]] x !llvm.array<1 x i64>
// CHECK-NEXT:    %[[GEP4:.+]] = llvm.getelementptr %[[ALLOCA_MALLOC4]][0, 0]
// CHECK-NEXT:    llvm.store %[[C1_I64]], %[[GEP4]]
// CHECK-NEXT:    %[[MALLOC4:.+]] = llvm.call @reactantXLAMalloc(%[[XDATA]], %[[C12]], %[[C1_I64]], %[[ALLOCA_MALLOC4]])
// CHECK-NEXT:    llvm.store %[[CF1_4]], %[[ALLOCA1]]
// CHECK-NEXT:    llvm.store %[[CF0]], %[[ALLOCA2]]
// CHECK-NEXT:    llvm.store %[[CF1]], %[[ALLOCA3]]
// CHECK-NEXT:    llvm.store %[[CF1]], %[[ALLOCA4]]
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[MALLOC1]], %[[ALLOCA1]], %[[C8_IDX]], %[[C1_I32]])
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[MALLOC2]], %[[ALLOCA2]], %[[C8_IDX]], %[[C1_I32]])
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[MALLOC3]], %[[ALLOCA3]], %[[C8_IDX]], %[[C1_I32]])
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[MALLOC4]], %[[ALLOCA4]], %[[C8_IDX]], %[[C1_I32]])
// CHECK-NEXT:    %[[GEP_EXEC1:.+]] = llvm.getelementptr %[[ALLOCA_EXEC]][0, 0]
// CHECK-NEXT:    llvm.store %[[CAST1]], %[[GEP_EXEC1]]
// CHECK-NEXT:    %[[GEP_EXEC2:.+]] = llvm.getelementptr %[[ALLOCA_EXEC]][0, 1]
// CHECK-NEXT:    llvm.store %[[CAST3]], %[[GEP_EXEC2]]
// CHECK-NEXT:    llvm.call @reactantXLAExec(%[[XDATA]], %{{.+}}, %[[C2_I64]], %[[ALLOCA_EXEC]])
// CHECK-NEXT:    %[[ZERO:.+]] = llvm.mlir.zero : i32
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[ALLOCA1]], %[[MALLOC1]], %[[C8_IDX]], %[[C2_I32]])
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[ALLOCA2]], %[[MALLOC2]], %[[C8_IDX]], %[[C2_I32]])
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[ALLOCA3]], %[[MALLOC3]], %[[C8_IDX]], %[[C2_I32]])
// CHECK-NEXT:    llvm.call @reactantXLAMemcpy(%[[XDATA]], %[[ALLOCA4]], %[[MALLOC4]], %[[C8_IDX]], %[[C2_I32]])
// CHECK-NEXT:    %[[RES1:.+]] = llvm.load %[[ALLOCA1]]
// CHECK-NEXT:    %[[RES3:.+]] = llvm.load %[[ALLOCA3]]
// CHECK-NEXT:    llvm.call @printf(%[[STR]], %[[RES1]], %[[RES3]])
// CHECK-NEXT:    %[[RES2:.+]] = llvm.load %[[ALLOCA2]]
// CHECK-NEXT:    %[[RES4:.+]] = llvm.load %[[ALLOCA4]]
// CHECK-NEXT:    llvm.call @printf(%[[STR]], %[[RES2]], %[[RES4]])
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
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
