// RUN: enzymexlamlir-opt %s --delinearize-indexing | FileCheck %s

// An access rebuilt on the buffer's own shape keeps the alignment it carried:
// it is the same address, and without it the access would be emitted at the
// element's ABI alignment.

#set = affine_set<(d0, d1) : (-d0 + 1 >= 0, d1 == 0)>
module {
  func.func @from_alloca(%v: f64) -> f64 {
    %a = memref.alloca() alignment = 8 : memref<2x3xf64>
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (3, 3, 3) {
      affine.if #set(%j, %k) {
        %p = "enzymexla.memref2pointer"(%a) : (memref<2x3xf64>) -> !llvm.ptr<3>
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64>
        affine.store %v, %m[%i + %j * 3] {alignment = 4 : i64} : memref<?xf64>
      }
    }
    %p = "enzymexla.memref2pointer"(%a) : (memref<2x3xf64>) -> !llvm.ptr<3>
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64>
    %r = affine.load %m[4] {alignment = 4 : i64} : memref<?xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @from_alloca(%arg0: f64) -> f64 {
// CHECK-NEXT:   %alloca = memref.alloca() alignment = 8 : memref<2x3xf64>
// CHECK-NEXT:   affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (3, 3, 3) {
// CHECK-NEXT:     affine.if #set(%arg2, %arg3) {
// CHECK-NEXT:       %3 = "enzymexla.memref2pointer"(%alloca) : (memref<2x3xf64>) -> !llvm.ptr<3>
// CHECK-NEXT:       %4 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr<3>) -> memref<2x3xf64>
// CHECK-NEXT:       affine.store %arg0, %4[%arg1 floordiv 3 + %arg2, %arg1 mod 3] {alignment = 4 : i64} : memref<2x3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = "enzymexla.memref2pointer"(%alloca) : (memref<2x3xf64>) -> !llvm.ptr<3>
// CHECK-NEXT:   %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<3>) -> memref<2x3xf64>
// CHECK-NEXT:   %2 = affine.load %1[1, 1] {alignment = 4 : i64} : memref<2x3xf64>
// CHECK-NEXT:   return %2 : f64
// CHECK-NEXT: }
