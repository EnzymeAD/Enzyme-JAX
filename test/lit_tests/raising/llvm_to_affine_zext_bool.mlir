// RUN: enzymexlamlir-opt --llvm-to-affine-access --split-input-file %s | FileCheck %s

// A slot chosen by a flag, `5 - zext(symmetric)` as MFEM's `symmetric ?
// op(4) : op(5)` folds to. A symbol is cast to index sign-extending, and an
// i1 `true` sign-extends to -1: read through the zero extension, the slot
// would be 6. The zero-extended value is the symbol instead.

llvm.func @zext_bool(%p: !llvm.ptr, %b: i1) {
  %c5 = arith.constant 5 : i32
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 4 {
    %z = arith.extui %b : i1 to i32
    %s = arith.subi %c5, %z : i32
    %e = arith.extsi %s : i32 to i64
    %g = llvm.getelementptr inbounds %p[%e] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
  }
  llvm.return
}

// CHECK: llvm.func @zext_bool(%arg0: !llvm.ptr, %arg1: i1) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_castui %arg1 : i1 to index
// CHECK-NEXT:   affine.for %arg2 = 0 to 4 {
// CHECK-NEXT:     %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:     affine.store %cst, %1[-symbol(%0) + 5] {ordering = 0 : i64} : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

// -----

// A zero extension of a loop variable, known non-negative, is still read
// through to the variable itself.

llvm.func @zext_iv(%p: !llvm.ptr) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 4 {
    %iv = arith.index_cast %i : index to i32
    %z = arith.extui %iv : i32 to i64
    %g = llvm.getelementptr inbounds %p[%z] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
  }
  llvm.return
}

// CHECK: llvm.func @zext_iv(%arg0: !llvm.ptr) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.for %arg1 = 0 to 4 {
// CHECK-NEXT:     %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:     affine.store %cst, %0[%arg1] {ordering = 0 : i64} : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

// -----

// A zero extension of a value the loop carries: nothing bounds it, so it is
// not read through (asking the loop's bounds about it read past them).

llvm.func @zext_carried(%p: !llvm.ptr) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %r = affine.for %i = 0 to 4 iter_args(%a = %c0) -> i32 {
    %z = arith.extui %a : i32 to i64
    %g = llvm.getelementptr inbounds %p[%z] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
    %n = arith.addi %a, %c1 : i32
    affine.yield %n : i32
  }
  llvm.return
}

// CHECK: llvm.func @zext_carried(%arg0: !llvm.ptr) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %c0_i32) -> (i32) {
// CHECK-NEXT:     %1 = arith.extui %arg2 : i32 to i64
// CHECK-NEXT:     %2 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:     %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:     memref.store %cst, %3[%c0] {ordering = 0 : i64} : memref<?xf64>
// CHECK-NEXT:     %4 = arith.addi %arg2, %c1_i32 : i32
// CHECK-NEXT:     affine.yield %4 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }
