// RUN: enzymexlamlir-opt --canonicalize-parallel %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @union_scratch
  // CHECK: %[[A:.+]] = memref.alloca() {alignment = 8 : i64} : memref<4xf64>
  // CHECK-NEXT: memref.store %{{.*}}, %[[A]][%{{.*}}] : memref<4xf64>
  // CHECK-NEXT: memref.load %[[A]][%{{.*}}] : memref<4xf64>
  // CHECK-NOT: memref2pointer
  func.func @union_scratch(%i: index, %v: f64) -> f64 {
    %a = memref.alloca() {alignment = 8 : i64} : memref<!llvm.struct<"union.anon", (array<2 x array<2 x f64>>)>>
    %p = "enzymexla.memref2pointer"(%a) : (memref<!llvm.struct<"union.anon", (array<2 x array<2 x f64>>)>>) -> !llvm.ptr
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    memref.store %v, %m[%i] : memref<?xf64>
    %r = memref.load %m[%i] : memref<?xf64>
    return %r : f64
  }

  // CHECK-LABEL: func.func @shared_array
  // CHECK: %[[A:.+]] = memref.alloca() : memref<24xi32, 3>
  // CHECK: "enzymexla.memref2pointer"(%[[A]]) : (memref<24xi32, 3>) -> !llvm.ptr<3>
  func.func @shared_array() -> !llvm.ptr<3> {
    %a = memref.alloca() : memref<3x!llvm.array<2 x array<4 x i32>>, 3>
    %p = "enzymexla.memref2pointer"(%a) : (memref<3x!llvm.array<2 x array<4 x i32>>, 3>) -> !llvm.ptr<3>
    return %p : !llvm.ptr<3>
  }

  // CHECK-LABEL: func.func @mixed_leaves
  // CHECK: memref.alloca() : memref<!llvm.struct<(i32, f64)>>
  func.func @mixed_leaves() -> !llvm.ptr {
    %a = memref.alloca() : memref<!llvm.struct<(i32, f64)>>
    %p = "enzymexla.memref2pointer"(%a) : (memref<!llvm.struct<(i32, f64)>>) -> !llvm.ptr
    return %p : !llvm.ptr
  }

  // CHECK-LABEL: func.func @padded
  // CHECK: memref.alloca() : memref<!llvm.struct<(i8, i32)>>
  func.func @padded() -> !llvm.ptr {
    %a = memref.alloca() : memref<!llvm.struct<(i8, i32)>>
    %p = "enzymexla.memref2pointer"(%a) : (memref<!llvm.struct<(i8, i32)>>) -> !llvm.ptr
    return %p : !llvm.ptr
  }

  // CHECK-LABEL: func.func @typed_user
  // CHECK: memref.alloca() : memref<!llvm.struct<(array<2 x f64>)>>
  func.func @typed_user(%i: index) -> !llvm.struct<(array<2 x f64>)> {
    %a = memref.alloca() : memref<!llvm.struct<(array<2 x f64>)>>
    %p = "enzymexla.memref2pointer"(%a) : (memref<!llvm.struct<(array<2 x f64>)>>) -> !llvm.ptr
    %r = memref.load %a[] : memref<!llvm.struct<(array<2 x f64>)>>
    return %r : !llvm.struct<(array<2 x f64>)>
  }
}
