// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access,canonicalize)" | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {

  func.func @test_conversion(%arg0: memref<100x10xf64, 1>, %arg1: memref<100x10xf64, 1>) -> (f64, f64) {
    // Convert memrefs to pointers
    %ptr0 = "enzymexla.memref2pointer"(%arg0) : (memref<100x10xf64, 1>) -> !llvm.ptr<1>
    %ptr1 = "enzymexla.memref2pointer"(%arg1) : (memref<100x10xf64, 1>) -> !llvm.ptr<1>

    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index

    %val2_ref = memref.alloca() : memref<f64>
    %val3_ref = memref.alloca() : memref<f64>

    // Put loads inside affine.parallel
    affine.parallel (%i) = (0) to (10) {
      %val1 = affine.load %arg0[%i*10,%i] : memref<100x10xf64,1>
      %val1_i64 = arith.fptosi %val1 : f64 to i64
      %i_i64 = arith.index_cast %i : index to i64
      %index1 = arith.addi %val1_i64, %i_i64 : i64
      %gep0 = llvm.getelementptr inbounds %ptr0[%index1] : (!llvm.ptr<1>,i64) -> !llvm.ptr<1>, f64
      %val2 = llvm.load %gep0 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %c8 = arith.constant 8 : i64
      %mul = arith.muli %i_i64, %c8 : i64
      %gep1 = llvm.getelementptr %ptr1[%mul] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %val3 = llvm.load %gep1 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
      %c99 = arith.constant 99 : index
      %is_last = arith.cmpi eq, %i, %c99 : index
      scf.if %is_last {
        memref.store %val2, %val2_ref[] : memref<f64>
        memref.store %val3, %val3_ref[] : memref<f64>
      }
    }

    %result0 = memref.load %val2_ref[] : memref<f64>
    %result1 = memref.load %val3_ref[] : memref<f64>
    return %result0, %result1 : f64, f64
  }
}

// CHECK-LABEL: test_conversion
// CHECK-NEXT : %c99 = arith.constant 99 : index
// CHECK-NEXT : %0 = "enzymexla.memref2pointer"(%arg0) : (memref<100x10xf64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT : %1 = "enzymexla.memref2pointer"(%arg1) : (memref<100x10xf64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT : %alloca = memref.alloca() : memref<f64>
// CHECK-NEXT : %alloca_0 = memref.alloca() : memref<f64>
// CHECK-NEXT : affine.parallel (%arg2) = (0) to (10) {
// CHECK-NEXT :   %4 = affine.load %arg0[%arg2 * 10, %arg2] : memref<100x10xf64, 1>
// CHECK-NEXT :   %5 = arith.fptosi %4 : f64 to i64
// CHECK-NEXT :   %6 = arith.index_cast %arg2 : index to i64
// CHECK-NEXT :   %7 = arith.addi %5, %6 : i64
// CHECK-NEXT :   %8 = arith.index_cast %7 : i64 to index
// CHECK-NEXT :   %9 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<1000xf64, 1>
// CHECK-NEXT :   %10 = memref.load %9[%8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<1000xf64, 1>
// CHECK-NEXT :   %11 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr<1>) -> memref<?xf64, 1>
// CHECK-NEXT :   %12 = affine.load %11[%arg2 * 8] {alignment = 8 : i64, ordering = 0 : i64} : memref<?xf64, 1>
// CHECK-NEXT :   %13 = arith.cmpi eq, %arg2, %c99 : index
// CHECK-NEXT :   scf.if %13 {
// CHECK-NEXT :     memref.store %10, %alloca[] : memref<f64>
// CHECK-NEXT :     memref.store %12, %alloca_0[] : memref<f64>
// CHECK-NEXT :   }
// CHECK-NEXT : }
// CHECK-NEXT : %2 = memref.load %alloca[] : memref<f64>
// CHECK-NEXT : %3 = memref.load %alloca_0[] : memref<f64>
// CHECK-NEXT : return %2, %3 : f64, f64
// CHECK-NEXT : }
