// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>>} {
  llvm.func @ptr_element(%base: !llvm.ptr, %off: i64) -> !llvm.ptr {
    %g = llvm.getelementptr inbounds|nuw %base[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %v = affine.load %m[0] : memref<?x!llvm.ptr>
    llvm.return %v : !llvm.ptr
  }

  llvm.func @float_element(%base: !llvm.ptr, %off: i64) -> f64 {
    %g = llvm.getelementptr inbounds|nuw %base[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    %v = affine.load %m[0] : memref<?xf64>
    llvm.return %v : f64
  }
}

// A view of pointers folds the same way a view of floats does.
// CHECK-LABEL: llvm.func @ptr_element(
// CHECK-NOT: llvm.getelementptr
// CHECK: "enzymexla.pointer2memref"(%arg0)

// CHECK-LABEL: llvm.func @float_element(
// CHECK-NOT: llvm.getelementptr
// CHECK: "enzymexla.pointer2memref"(%arg0)
