// RUN: enzymexlamlir-opt %s --delinearize-indexing | FileCheck %s

module {
  func.func @main(%arg0: memref<i64>, %i: index) -> f32 {
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<i64>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<10xf32>
    %2 = affine.load %1[%i] : memref<10xf32>
    return %2 : f32
  }
}

// CHECK:  func.func @main(%arg0: memref<i64>, %arg1: index) -> f32 {
// CHECK-NEXT:    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<i64>) -> !llvm.ptr<1>
// CHECK-NEXT:    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<10xf32>
// CHECK-NEXT:    %2 = affine.load %1[%arg1] : memref<10xf32>
// CHECK-NEXT:    return %2 : f32
// CHECK-NEXT:  }
