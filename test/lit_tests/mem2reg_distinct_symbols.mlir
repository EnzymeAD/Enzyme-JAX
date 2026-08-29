// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// Two accesses whose affine maps agree land on the same slot only when the
// symbols bound to the map agree as well; a load indexed by a different
// symbol names a different location and must not be forwarded.
func.func @distinct_symbols(%i: index, %s1: index, %s2: index, %v: f64) -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<81 x f64> : (i32) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?xf64>
  affine.store %v, %view[%i + symbol(%s1) * 9] : memref<?xf64>
  %a = affine.load %view[%i + symbol(%s1) * 9] : memref<?xf64>
  %b = affine.load %view[%i + symbol(%s2) * 9] : memref<?xf64>
  %r = arith.addf %a, %b : f64
  return %r : f64
}

// The store and the first load bind the same symbol, so that load takes the
// stored value; the second load binds another symbol and stays a load.
// CHECK-LABEL: func.func @distinct_symbols(
// CHECK-SAME: %[[I:[a-z0-9]+]]: index, %[[S1:[a-z0-9]+]]: index, %[[S2:[a-z0-9]+]]: index, %[[V:[a-z0-9]+]]: f64
// CHECK: affine.store %[[V]], %{{.*}}[%[[I]] + symbol(%[[S1]]) * 9]
// CHECK: %[[B:.+]] = affine.load %{{.*}}[%[[I]] + symbol(%[[S2]]) * 9]
// CHECK: %[[R:.+]] = arith.addf %[[V]], %[[B]] : f64
// CHECK: return %[[R]] : f64
