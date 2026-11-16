// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// The accesses of the fields that replace an access of an aggregate keep what
// it said of its alignment, less each field's offset: a complex stored at an
// alignment of 4 is two reals stored at 4.

func.func @real_to_complex(%out: memref<4x2xf64, 1>, %in: memref<4xf64, 1>) -> f64 {
  %zero = arith.constant 0.000000e+00 : f64
  %c = llvm.mlir.constant(dense<0.000000e+00> : tensor<2xf64>) : !llvm.array<2 x f64>
  %p = "enzymexla.memref2pointer"(%out) : (memref<4x2xf64, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (4) {
    %x = affine.load %in[%i] : memref<4xf64, 1>
    %a = llvm.insertvalue %x, %c[0] : !llvm.array<2 x f64>
    %b = llvm.insertvalue %zero, %a[1] : !llvm.array<2 x f64>
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f64>, 1>
    affine.store %b, %v[%i] {alignment = 4 : i64} : memref<?x!llvm.array<2 x f64>, 1>
  }
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f64>, 1>
  %w = affine.load %v[1] {alignment = 4 : i64} : memref<?x!llvm.array<2 x f64>, 1>
  %r = llvm.extractvalue %w[1] : !llvm.array<2 x f64>
  return %r : f64
}

// CHECK-LABEL: func.func @real_to_complex(%arg0: memref<4x2xf64, 1>, %arg1: memref<4xf64, 1>) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.parallel (%arg2) = (0) to (4) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg2] : memref<4xf64, 1>
// CHECK-NEXT:     affine.store %1, %arg0[%arg2, 0] {alignment = 4 : i64} : memref<4x2xf64, 1>
// CHECK-NEXT:     affine.store %cst, %arg0[%arg2, 1] {alignment = 4 : i64} : memref<4x2xf64, 1>
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = affine.load %arg0[1, 1] {alignment = 4 : i64} : memref<4x2xf64, 1>
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }
