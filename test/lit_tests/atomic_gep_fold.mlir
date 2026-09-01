// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

// The gep an access is taken through folds into its index, for an atomic as
// for a load or a store: the view is then taken of the base, which is what
// lets a kernel's pointer argument have nothing but a view reading it.

module {
  func.func @enzyme_atomic(%p: !llvm.ptr, %i: i64, %v: f64) {
    %g = llvm.getelementptr inbounds %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    %c0 = arith.constant 0 : index
    %old = enzyme.atomic_rmw addf %v, %m[%c0] monotonic {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
    return
  }

  func.func @memref_atomic(%p: !llvm.ptr, %i: i64, %v: i32) {
    %g = llvm.getelementptr inbounds %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi32>
    %c0 = arith.constant 0 : index
    %old = memref.atomic_rmw addi %v, %m[%c0] : (i32, memref<?xi32>) -> i32
    return
  }

  // an element type the view does not share with the gep is left alone
  func.func @element_type_differs(%p: !llvm.ptr, %i: i64, %v: f64) {
    %g = llvm.getelementptr inbounds %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    %c0 = arith.constant 0 : index
    %old = enzyme.atomic_rmw addf %v, %m[%c0] monotonic : (f64, memref<?xf64>) -> f64
    return
  }
}

// CHECK:  func.func @enzyme_atomic(%[[a1:.+]]: !llvm.ptr, %[[a2:.+]]: i64, %[[a3:.+]]: f64) {
// CHECK-NEXT:  %[[a4:.+]] = "enzymexla.pointer2memref"(%[[a1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[a5:.+]] = arith.index_cast %[[a2]] : i64 to index
// CHECK-NEXT:  %[[a6:.+]] = enzyme.atomic_rmw addf %[[a3]], %[[a4]][%[[a5]]] monotonic {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @memref_atomic(%[[a1:.+]]: !llvm.ptr, %[[a2:.+]]: i64, %[[a3:.+]]: i32) {
// CHECK-NEXT:  %[[a4:.+]] = "enzymexla.pointer2memref"(%[[a1]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:  %[[a5:.+]] = arith.index_cast %[[a2]] : i64 to index
// CHECK-NEXT:  %[[a6:.+]] = memref.atomic_rmw addi %[[a3]], %[[a4]][%[[a5]]] : (i32, memref<?xi32>) -> i32
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @element_type_differs(%[[a1:.+]]: !llvm.ptr, %[[a2:.+]]: i64, %[[a3:.+]]: f64) {
// CHECK-NEXT:  %[[a4:.+]] = arith.constant 0 : index
// CHECK-NEXT:  %[[a5:.+]] = llvm.getelementptr inbounds %[[a1]][%[[a2]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
// CHECK-NEXT:  %[[a6:.+]] = "enzymexla.pointer2memref"(%[[a5]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[a7:.+]] = enzyme.atomic_rmw addf %[[a3]], %[[a6]][%[[a4]]] monotonic : (f64, memref<?xf64>) -> f64
// CHECK-NEXT:  return
// CHECK-NEXT:  }
