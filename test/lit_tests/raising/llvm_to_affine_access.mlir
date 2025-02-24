// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

module {
  func.func @rewrite(%arg : !llvm.ptr, %cst : f64) {
    %c40 = arith.constant 40 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (72, 256) {
      %3 = arith.muli %arg4, %c40 : index
      %4 = arith.index_castui %3 : index to i64
      %5 = arith.index_castui %arg3 : index to i64
      %6 = arith.addi %4, %5 : i64
      %28 = llvm.getelementptr inbounds %arg[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %cst, %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    }
    return
  }
}

// CHECK-LABEL: rewrite
// CHECK:    affine.parallel (%arg2, %arg3) = (0, 0) to (72, 256) {
// CHECK-NEXT:      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:      affine.store %arg1, %0[%arg3 * 40 + %arg2] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64>
// CHECK-NEXT:    }

// -----

module {
  func.func @i8gep(%arg : !llvm.ptr, %cst : f64) {
    %c40 = arith.constant 40 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (72, 256) {
      %3 = arith.muli %arg4, %c40 : index
      %4 = arith.index_castui %3 : index to i64
      %28 = llvm.getelementptr inbounds %arg[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %cst, %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    }
    return
  }
}

// CHECK-LABEL: i8gep
// CHECK:      affine.parallel (%arg2, %arg3) = (0, 0) to (72, 256) {
// CHECK-NEXT:        %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:        affine.store %arg1, %0[%arg3 * 5] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64>
// CHECK-NEXT:      }
