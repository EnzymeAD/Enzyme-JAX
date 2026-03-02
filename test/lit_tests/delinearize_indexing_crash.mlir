
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
    memref.store %val, %1[%c0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
    return
  }
}
