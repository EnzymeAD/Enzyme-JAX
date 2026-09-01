// RUN: enzymexlamlir-opt %s --delinearize-indexing | FileCheck %s

// A flat view of a buffer is rebuilt on the shape the buffer was declared
// with, whether that buffer is a function's argument or an alloca beside the
// accesses. MFEM's shared-memory scratch is the latter, and left flat it
// reaches the raiser as a view that is stored through.

#set = affine_set<(d0, d1) : (-d0 + 1 >= 0, d1 == 0)>
module {

  func.func @from_alloca(%v: f64) {
    %a = memref.alloca() {alignment = 8 : i64} : memref<2x3xf64>
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (3, 3, 3) {
      affine.if #set(%j, %k) {
        %p = "enzymexla.memref2pointer"(%a) : (memref<2x3xf64>) -> !llvm.ptr<3>
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64>
        affine.store %v, %m[%i + %j * 3] : memref<?xf64>
      }
    }
    return
  }

  func.func @from_arg(%a: memref<2x3xf64>, %v: f64) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (3, 3, 3) {
      affine.if #set(%j, %k) {
        %p = "enzymexla.memref2pointer"(%a) : (memref<2x3xf64>) -> !llvm.ptr<3>
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64>
        affine.store %v, %m[%i + %j * 3] : memref<?xf64>
      }
    }
    return
  }

  // a load and a store of the same view, over three dimensions
  func.func @three_dims(%v: f64) -> f64 {
    %a = memref.alloca() : memref<2x3x4xf64>
    %p = "enzymexla.memref2pointer"(%a) : (memref<2x3x4xf64>) -> !llvm.ptr
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    affine.store %v, %m[7] : memref<?xf64>
    %r = affine.load %m[13] : memref<?xf64>
    return %r : f64
  }

  // an element type the view does not share with its buffer is left alone
  func.func @element_type_differs(%v: i32) {
    %a = memref.alloca() : memref<2x3xf64>
    %p = "enzymexla.memref2pointer"(%a) : (memref<2x3xf64>) -> !llvm.ptr
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xi32>
    affine.store %v, %m[5] : memref<?xi32>
    return
  }

  // a heap buffer
  func.func @from_alloc(%v: f64) {
    %a = memref.alloc() : memref<2x3xf64>
    %p = "enzymexla.memref2pointer"(%a) : (memref<2x3xf64>) -> !llvm.ptr
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
    affine.store %v, %m[4] : memref<?xf64>
    return
  }

  // a buffer that arrives as a block argument of a loop, not of the function
  func.func @from_loop_arg(%a: memref<2x3xf64>, %b: memref<2x3xf64>, %v: f64) -> memref<2x3xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%buf = %a) -> memref<2x3xf64> {
      %p = "enzymexla.memref2pointer"(%buf) : (memref<2x3xf64>) -> !llvm.ptr
      %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
      affine.store %v, %m[5] : memref<?xf64>
      scf.yield %b : memref<2x3xf64>
    }
    return %r : memref<2x3xf64>
  }

}

// CHECK:  func.func @from_alloca(%[[v1:.+]]: f64) {
// CHECK-NEXT:  %[[v2:.+]] = memref.alloca() {alignment = 8 : i64} : memref<2x3xf64>
// CHECK-NEXT:  affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (3, 3, 3) {
// CHECK-NEXT:  affine.if #set(%arg2, %arg3) {
// CHECK-NEXT:  %[[v3:.+]] = "enzymexla.memref2pointer"(%[[v2]]) : (memref<2x3xf64>) -> !llvm.ptr<3>
// CHECK-NEXT:  %[[v4:.+]] = "enzymexla.pointer2memref"(%[[v3]]) : (!llvm.ptr<3>) -> memref<2x3xf64>
// CHECK-NEXT:  affine.store %[[v1]], %[[v4]][%arg1 floordiv 3 + %arg2, %arg1 mod 3] : memref<2x3xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @from_arg(%[[v1:.+]]: memref<2x3xf64>, %[[v2:.+]]: f64) {
// CHECK-NEXT:  affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (3, 3, 3) {
// CHECK-NEXT:  affine.if #set(%arg3, %arg4) {
// CHECK-NEXT:  %[[v3:.+]] = "enzymexla.memref2pointer"(%[[v1]]) : (memref<2x3xf64>) -> !llvm.ptr<3>
// CHECK-NEXT:  %[[v4:.+]] = "enzymexla.pointer2memref"(%[[v3]]) : (!llvm.ptr<3>) -> memref<2x3xf64>
// CHECK-NEXT:  affine.store %[[v2]], %[[v4]][%arg2 floordiv 3 + %arg3, %arg2 mod 3] : memref<2x3xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @three_dims(%[[v1:.+]]: f64) -> f64 {
// CHECK-NEXT:  %[[v2:.+]] = memref.alloca() : memref<2x3x4xf64>
// CHECK-NEXT:  %[[v3:.+]] = "enzymexla.memref2pointer"(%[[v2]]) : (memref<2x3x4xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[v4:.+]] = "enzymexla.pointer2memref"(%[[v3]]) : (!llvm.ptr) -> memref<2x3x4xf64>
// CHECK-NEXT:  affine.store %[[v1]], %[[v4]][0, 1, 3] : memref<2x3x4xf64>
// CHECK-NEXT:  %[[v5:.+]] = affine.load %[[v4]][1, 0, 1] : memref<2x3x4xf64>
// CHECK-NEXT:  return %[[v5]] : f64
// CHECK-NEXT:  }

// CHECK:  func.func @element_type_differs(%[[v1:.+]]: i32) {
// CHECK-NEXT:  %[[v2:.+]] = memref.alloca() : memref<2x3xf64>
// CHECK-NEXT:  %[[v3:.+]] = "enzymexla.memref2pointer"(%[[v2]]) : (memref<2x3xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[v4:.+]] = "enzymexla.pointer2memref"(%[[v3]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:  affine.store %[[v1]], %[[v4]][5] : memref<?xi32>
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @from_alloc(%[[w1:.+]]: f64) {
// CHECK-NEXT:  %[[w2:.+]] = memref.alloc() : memref<2x3xf64>
// CHECK-NEXT:  %[[w3:.+]] = "enzymexla.memref2pointer"(%[[w2]]) : (memref<2x3xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[w4:.+]] = "enzymexla.pointer2memref"(%[[w3]]) : (!llvm.ptr) -> memref<2x3xf64>
// CHECK-NEXT:  affine.store %[[w1]], %[[w4]][1, 1] : memref<2x3xf64>
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @from_loop_arg(%[[w1:.+]]: memref<2x3xf64>, %[[w2:.+]]: memref<2x3xf64>, %[[w3:.+]]: f64) -> memref<2x3xf64> {
// CHECK-NEXT:  %[[w4:.+]] = arith.constant 0 : index
// CHECK-NEXT:  %[[w5:.+]] = arith.constant 1 : index
// CHECK-NEXT:  %[[w6:.+]] = arith.constant 2 : index
// CHECK-NEXT:  %[[w7:.+]] = scf.for %arg3 = %[[w4]] to %[[w6]] step %[[w5]] iter_args(%arg4 = %[[w1]]) -> (memref<2x3xf64>) {
// CHECK-NEXT:  %[[w8:.+]] = "enzymexla.memref2pointer"(%arg4) : (memref<2x3xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[w9:.+]] = "enzymexla.pointer2memref"(%[[w8]]) : (!llvm.ptr) -> memref<2x3xf64>
// CHECK-NEXT:  affine.store %[[w3]], %[[w9]][1, 2] : memref<2x3xf64>
// CHECK-NEXT:  scf.yield %[[w2]] : memref<2x3xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[w7]] : memref<2x3xf64>
// CHECK-NEXT:  }
