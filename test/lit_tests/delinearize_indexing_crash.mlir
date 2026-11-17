
// RUN: enzymexlamlir-opt --delinearize-indexing %s | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  // CHECK-LABEL: func.func @crash
  func.func @crash(%arg0: memref<1x10x10xf64, 1>) {
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<1x10x10xf64, 1>) -> !llvm.ptr<1>
    %c0 = arith.constant 0 : index
    %val = arith.constant 1.0 : f64
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
    
    // Check that we can transform this without crashing
    // CHECK: memref.store
    memref.store %val, %1[%c0] alignment(8) {ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
    return
  }

  // Scalar (0-rank) pointer2memref view onto a static multi-dimensional buffer.
  // CHECK-LABEL: func.func @scalar_view_to_multidim
  func.func @scalar_view_to_multidim(%arg0: memref<4x8xf64, 1>) -> f64 {
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<4x8xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<f64, 1>
    // CHECK: affine.load %{{.+}}[0, 0] : memref<4x8xf64, 1>
    %2 = affine.load %1[] : memref<f64, 1>
    %3 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<f64, 1>
    // CHECK: memref.store %{{.+}}, %{{.+}}[%c0{{.*}}, %c0{{.*}}] : memref<4x8xf64, 1>
    memref.store %2, %3[] : memref<f64, 1>
    return %2 : f64
  }

  // Zero-sized dimension in target buffer must not divide/modulo by zero.
  // CHECK-LABEL: func.func @zero_sized_dim_no_crash
  func.func @zero_sized_dim_no_crash(%arg0: memref<4x0xf64, 1>, %idx: index) -> f64 {
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<4x0xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xf64, 1>
    // CHECK: memref.load %{{.+}}[%arg1] : memref<?xf64, 1>
    %2 = memref.load %1[%idx] : memref<?xf64, 1>
    return %2 : f64
  }

  // Multi-rank pointer2memref view must not be treated as a 1-D flat index.
  // CHECK-LABEL: func.func @multirank_view_no_crash
  func.func @multirank_view_no_crash(%arg0: memref<4x8xf64, 1>, %i: index, %j: index) -> f64 {
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<4x8xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<2x16xf64, 1>
    // CHECK: memref.load %{{.+}}[%arg1, %arg2] : memref<2x16xf64, 1>
    %2 = memref.load %1[%i, %j] : memref<2x16xf64, 1>
    return %2 : f64
  }
}
