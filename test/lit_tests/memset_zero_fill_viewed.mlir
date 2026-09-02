// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

module {
  func.func @chosen(%a: !llvm.ptr, %b: !llvm.ptr, %c: i1, %i: i64) -> f64 {
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %ga = llvm.getelementptr %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %gb = llvm.getelementptr %b[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %s = arith.select %c, %ga, %gb : !llvm.ptr
    "llvm.intr.memset"(%s, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %va = llvm.load %a : !llvm.ptr -> f64
    %vb = llvm.load %b : !llvm.ptr -> f64
    %r = arith.addf %va, %vb : f64
    return %r : f64
  }
  func.func @punned(%a: !llvm.ptr) -> f64 {
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    "llvm.intr.memset"(%a, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %va = llvm.load %a : !llvm.ptr -> f64
    %g = llvm.getelementptr %a[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %vi = llvm.load %g : !llvm.ptr -> i32
    %vf = arith.sitofp %vi : i32 to f64
    %r = arith.addf %va, %vf : f64
    return %r : f64
  }
  func.func @unviewed(%a: !llvm.ptr) {
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    "llvm.intr.memset"(%a, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
}

// CHECK-LABEL:  func.func @chosen(
// CHECK-NOT:  llvm.intr.memset
// CHECK:  %[[sel:.+]] = arith.select %arg2, %{{.*}}, %{{.*}} : memref<?xf64>
// CHECK-NEXT:  affine.for %[[iv:.+]] = 0 to 3 {
// CHECK-NEXT:  affine.store %{{.*}}, %[[sel]][%[[iv]]] : memref<?xf64>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @punned(
// CHECK:  llvm.intr.memset

// CHECK-LABEL:  func.func @unviewed(
// CHECK:  llvm.intr.memset
