// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// A capture built one field at a time and read back whole: the load of the
// aggregate has no single store to forward from, but each field taken out of
// it does.

module {
  llvm.func @use(!llvm.ptr, !llvm.ptr)
  func.func @affine_capture(%a : !llvm.ptr, %b : !llvm.ptr) {
    %c1 = arith.constant 1 : i32
    %alloca = llvm.alloca %c1 x !llvm.struct<"cap", (ptr, ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %f = "enzymexla.pointer2memref"(%alloca) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    affine.store %a, %f[0] : memref<?x!llvm.ptr>
    affine.store %b, %f[1] : memref<?x!llvm.ptr>
    %s = "enzymexla.pointer2memref"(%alloca) : (!llvm.ptr) -> memref<?x!llvm.struct<"cap", (ptr, ptr)>>
    %agg = affine.load %s[0] : memref<?x!llvm.struct<"cap", (ptr, ptr)>>
    %x = llvm.extractvalue %agg[0] : !llvm.struct<"cap", (ptr, ptr)>
    %y = llvm.extractvalue %agg[1] : !llvm.struct<"cap", (ptr, ptr)>
    llvm.call @use(%x, %y) : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @affine_capture(
// CHECK-SAME:      %[[a:.*]]: !llvm.ptr, %[[b:.*]]: !llvm.ptr) {
// CHECK-NOT:       llvm.extractvalue
// CHECK:           llvm.call @use(%[[a]], %[[b]])
// CHECK:           return

// -----

module {
  llvm.func @use(!llvm.ptr, !llvm.ptr)
  func.func @llvm_capture(%a : !llvm.ptr, %b : !llvm.ptr) {
    %c1 = arith.constant 1 : i32
    %alloca = llvm.alloca %c1 x !llvm.struct<"cap", (ptr, ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %p0 = llvm.getelementptr %alloca[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %a, %p0 : !llvm.ptr, !llvm.ptr
    %p1 = llvm.getelementptr %alloca[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %b, %p1 : !llvm.ptr, !llvm.ptr
    %agg = llvm.load %alloca : !llvm.ptr -> !llvm.struct<"cap", (ptr, ptr)>
    %x = llvm.extractvalue %agg[0] : !llvm.struct<"cap", (ptr, ptr)>
    %y = llvm.extractvalue %agg[1] : !llvm.struct<"cap", (ptr, ptr)>
    llvm.call @use(%x, %y) : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @llvm_capture(
// CHECK-SAME:      %[[a:.*]]: !llvm.ptr, %[[b:.*]]: !llvm.ptr) {
// CHECK-NOT:       llvm.extractvalue
// CHECK:           llvm.call @use(%[[a]], %[[b]])
// CHECK:           return
