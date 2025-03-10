// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access,canonicalize)" | FileCheck %s

module {
  func.func @memarg_nonaffine_mul(%arg0 : memref<1000xf64>, %cst : f64) {
    %c40 = arith.constant 40 : index
    %arg = "enzymexla.memref2pointer"(%arg0) : (memref<1000xf64>) -> !llvm.ptr
    affine.parallel (%arg3, %arg4) = (0, 0) to (72, 256) {
      %3 = arith.muli %arg4, %c40 : index
      %4 = arith.index_castui %3 : index to i64
      %5 = arith.index_castui %arg3 : index to i64
      %6 = arith.muli %4, %5 : i64
      %28 = llvm.getelementptr inbounds %arg[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %cst, %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    }
    return
  }
}

// CHECK-LABEL: memarg_nonaffine_mul
// CHECK: %0 = "enzymexla.memref2pointer"(%arg0) : (memref<1000xf64>) -> !llvm.ptr
// CHECK-NEXT: affine.parallel (%arg2, %arg3) = (0, 0) to (72, 256) {
// CHECK-NEXT:   %1 = arith.muli %arg3, %c40 : index
// CHECK-NEXT:   %2 = arith.index_castui %1 : index to i64
// CHECK-NEXT:   %3 = arith.index_castui %arg2 : index to i64
// CHECK-NEXT:   %4 = arith.muli %2, %3 : i64
// CHECK-NEXT:   %5 = llvm.getelementptr inbounds %0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:   llvm.store %arg1, %5 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT: }
