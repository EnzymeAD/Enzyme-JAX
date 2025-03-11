// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(delinearize-indexing)" | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func @test_conversion(%arg0: memref<100x10xf64, 1>, %arg1: memref<100x10xf64, 1>) -> (f64, f64) {
    %c99 = arith.constant 99 : index
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<100x10xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.memref2pointer"(%arg1) : (memref<100x10xf64, 1>) -> !llvm.ptr<1>
    %alloca = memref.alloca() : memref<f64>
    %alloca_0 = memref.alloca() : memref<f64>
    affine.parallel (%arg2) = (0) to (10) {
      %4 = affine.load %arg0[%arg2 * 10, %arg2] : memref<100x10xf64, 1>
      %5 = arith.fptosi %4 : f64 to i64
      %6 = arith.index_cast %arg2 : index to i64
      %7 = arith.addi %5, %6 : i64
      %8 = arith.index_cast %7 : i64 to index
      %9 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<1000xf64, 1>
      %10 = memref.load %9[%8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<1000xf64, 1>
      %11 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr<1>) -> memref<?xf64, 1>
      %12 = affine.load %11[%arg2 * 8] {alignment = 8 : i64, ordering = 0 : i64} : memref<?xf64, 1>
      %13 = arith.cmpi eq, %arg2, %c99 : index
      scf.if %13 {
        memref.store %10, %alloca[] : memref<f64>
        memref.store %12, %alloca_0[] : memref<f64>
      }
    }
    %2 = memref.load %alloca[] : memref<f64>
    %3 = memref.load %alloca_0[] : memref<f64>
    return %2, %3 : f64, f64
  }
}

