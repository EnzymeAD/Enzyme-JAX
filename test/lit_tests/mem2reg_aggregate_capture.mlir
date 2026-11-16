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

// -----

// Promoting %dst materializes an extractvalue on the aggregate loaded from
// %src; promoting %src in the same pass iteration must order that extractvalue
// after the overwriting store in ^bb1 rather than forwarding %v1 from entry.

module {
  llvm.func @use_ptr(!llvm.ptr)
  func.func @chained_aggregate_forward(%v1 : !llvm.ptr, %v2 : !llvm.ptr, %v3 : !llvm.ptr) {
    %c1 = arith.constant 1 : i32
    %dst = llvm.alloca %c1 x !llvm.struct<"cap", (ptr, ptr)> : (i32) -> !llvm.ptr
    %src = llvm.alloca %c1 x !llvm.struct<"cap", (ptr, ptr)> : (i32) -> !llvm.ptr
    %src0 = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %v1, %src0 : !llvm.ptr, !llvm.ptr
    %src1 = llvm.getelementptr %src[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %v2, %src1 : !llvm.ptr, !llvm.ptr
    cf.br ^bb1
  ^bb1:
    llvm.store %v3, %src0 : !llvm.ptr, !llvm.ptr
    %agg = llvm.load %src : !llvm.ptr -> !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %agg, %dst : !llvm.struct<"cap", (ptr, ptr)>, !llvm.ptr
    %dst0 = llvm.getelementptr %dst[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    %res = llvm.load %dst0 : !llvm.ptr -> !llvm.ptr
    llvm.call @use_ptr(%res) : (!llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @chained_aggregate_forward(
// CHECK-SAME:      %{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr, %[[V3:.+]]: !llvm.ptr) {
// CHECK-NOT:       llvm.extractvalue
// CHECK:           llvm.call @use_ptr(%[[V3]])
// CHECK:           return

// -----

module {
  memref.global "private" @dst : memref<1x!llvm.struct<"cap", (ptr, ptr)>>
  memref.global "private" @src : memref<1x!llvm.struct<"cap", (ptr, ptr)>>
  llvm.func @use(!llvm.ptr, !llvm.ptr)
  func.func @captured_global_aggregate_forward(%v1 : !llvm.ptr, %v2 : !llvm.ptr) {
    %dst_mem = memref.get_global @dst : memref<1x!llvm.struct<"cap", (ptr, ptr)>>
    %src_mem = memref.get_global @src : memref<1x!llvm.struct<"cap", (ptr, ptr)>>
    %dst = "enzymexla.memref2pointer"(%dst_mem) : (memref<1x!llvm.struct<"cap", (ptr, ptr)>>) -> !llvm.ptr
    %src = "enzymexla.memref2pointer"(%src_mem) : (memref<1x!llvm.struct<"cap", (ptr, ptr)>>) -> !llvm.ptr
    %src1 = llvm.getelementptr %src[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %v2, %src1 : !llvm.ptr, !llvm.ptr
    %src0 = llvm.getelementptr %src[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %v1, %src0 : !llvm.ptr, !llvm.ptr
    %agg = llvm.load %src : !llvm.ptr -> !llvm.struct<"cap", (ptr, ptr)>
    llvm.store %agg, %dst : !llvm.struct<"cap", (ptr, ptr)>, !llvm.ptr
    %dst0 = llvm.getelementptr %dst[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"cap", (ptr, ptr)>
    %res1 = llvm.load %dst0 : !llvm.ptr -> !llvm.ptr
    %res2 = llvm.load %src0 : !llvm.ptr -> !llvm.ptr
    llvm.call @use(%res1, %res2) : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @captured_global_aggregate_forward(
// CHECK-SAME:      %[[V1:.+]]: !llvm.ptr, %{{.+}}: !llvm.ptr) {
// CHECK-NOT:       llvm.extractvalue
// CHECK:           %[[LOAD:.+]] = llvm.load %{{.+}} : !llvm.ptr -> !llvm.ptr
// CHECK:           llvm.call @use(%[[V1]], %[[LOAD]])
// CHECK:           return
