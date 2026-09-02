// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

// The gep a view is taken through folds into the access index for the affine
// atomic as for every other access, leaving the view on the gep's own base.
// As the affine load and store do, it leaves the non-affine form, carrying
// the kind, ordering and alignment the op named.

module {
  func.func @affine_atomic(%p: !llvm.ptr, %i: i64, %j: index, %v: f64) {
    %g = llvm.getelementptr inbounds %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    %old = enzyme.affine_atomic_rmw addf %v, %m, (affine_map<(d0) -> (d0)>)[%j] seq_cst {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
    return
  }

  // an element type the view does not share with the gep is left alone
  func.func @element_type_differs(%p: !llvm.ptr, %i: i64, %j: index, %v: f64) {
    %g = llvm.getelementptr inbounds %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    %old = enzyme.affine_atomic_rmw addf %v, %m, (affine_map<(d0) -> (d0)>)[%j] monotonic : (f64, memref<?xf64>) -> f64
    return
  }
}

// CHECK:  func.func @affine_atomic(%[[g1:.+]]: !llvm.ptr, %[[g2:.+]]: i64, %[[g3:.+]]: index, %[[g4:.+]]: f64) {
// CHECK-NEXT:  %[[g5:.+]] = "enzymexla.pointer2memref"(%[[g1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[g6:.+]] = arith.index_cast %[[g2]] : i64 to index
// CHECK-NEXT:  %[[g7:.+]] = arith.addi %[[g3]], %[[g6]] : index
// CHECK-NEXT:  %[[g8:.+]] = enzyme.atomic_rmw addf %[[g4]], %[[g5]][%[[g7]]] seq_cst {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @element_type_differs(%[[g1:.+]]: !llvm.ptr, %[[g2:.+]]: i64, %[[g3:.+]]: index, %[[g4:.+]]: f64) {
// CHECK-NEXT:  %[[g5:.+]] = llvm.getelementptr inbounds %[[g1]][%[[g2]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
// CHECK-NEXT:  %[[g6:.+]] = "enzymexla.pointer2memref"(%[[g5]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[g7:.+]] = enzyme.affine_atomic_rmw addf %[[g4]], %[[g6]], (#map) [%[[g3]]] monotonic : (f64, memref<?xf64>) -> f64
// CHECK-NEXT:  return
// CHECK-NEXT:  }
