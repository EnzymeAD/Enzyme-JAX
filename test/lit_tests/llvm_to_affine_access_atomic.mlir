// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// An atomic read-modify-write becomes an access on a view of its pointer,
// beside the loads and stores this pass rewrites. The enzyme atomic rather
// than the memref one: it is the only form with somewhere to put the
// ordering, and the alignment, that the llvm op carried.

module {
  func.func @fadd(%p: !llvm.ptr, %v: f64) {
    %old = llvm.atomicrmw fadd %p, %v monotonic {alignment = 8 : i64} : !llvm.ptr, f64
    return
  }

  func.func @add_seq_cst(%p: !llvm.ptr, %v: i32) {
    %old = llvm.atomicrmw add %p, %v seq_cst : !llvm.ptr, i32
    return
  }

  func.func @umax_in_shared(%p: !llvm.ptr<3>, %v: i32) {
    %old = llvm.atomicrmw umax %p, %v acquire : !llvm.ptr<3>, i32
    return
  }

  // a kind no arith one names is left alone
  func.func @nand(%p: !llvm.ptr, %v: i32) {
    %old = llvm.atomicrmw _xor %p, %v monotonic : !llvm.ptr, i32
    return
  }
  // the address arithmetic joins the index, as it does for a load
  func.func @through_gep(%p: !llvm.ptr, %i: i64, %v: f64) {
    %g = llvm.getelementptr inbounds %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %old = llvm.atomicrmw fadd %g, %v monotonic {alignment = 8 : i64} : !llvm.ptr, f64
    return
  }

}

// CHECK:  func.func @fadd(%[[c1:.+]]: !llvm.ptr, %[[c2:.+]]: f64) {
// CHECK-NEXT:  %[[c3:.+]] = "enzymexla.pointer2memref"(%[[c1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[c4:.+]] = enzyme.affine_atomic_rmw addf %[[c2]], %[[c3]], (#map) [] monotonic {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @add_seq_cst(%[[c1:.+]]: !llvm.ptr, %[[c2:.+]]: i32) {
// CHECK-NEXT:  %[[c3:.+]] = "enzymexla.pointer2memref"(%[[c1]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:  %[[c4:.+]] = enzyme.affine_atomic_rmw addi %[[c2]], %[[c3]], (#map) [] seq_cst : (i32, memref<?xi32>) -> i32
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @umax_in_shared(%[[c1:.+]]: !llvm.ptr<3>, %[[c2:.+]]: i32) {
// CHECK-NEXT:  %[[c3:.+]] = "enzymexla.pointer2memref"(%[[c1]]) : (!llvm.ptr<3>) -> memref<?xi32, 3>
// CHECK-NEXT:  %[[c4:.+]] = enzyme.affine_atomic_rmw maxu %[[c2]], %[[c3]], (#map) [] acquire : (i32, memref<?xi32, 3>) -> i32
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @nand(%[[c1:.+]]: !llvm.ptr, %[[c2:.+]]: i32) {
// CHECK-NEXT:  %[[c3:.+]] = llvm.atomicrmw _xor %[[c1]], %[[c2]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @through_gep(%[[c1:.+]]: !llvm.ptr, %[[c2:.+]]: i64, %[[c3:.+]]: f64) {
// CHECK-NEXT:  %[[c4:.+]] = arith.index_cast %[[c2]] : i64 to index
// CHECK-NEXT:  %[[c5:.+]] = "enzymexla.pointer2memref"(%[[c1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[c6:.+]] = enzyme.affine_atomic_rmw addf %[[c3]], %[[c5]], (#map1) [%[[c4]]] monotonic {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
// CHECK-NEXT:  return
// CHECK-NEXT:  }
