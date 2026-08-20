// RUN: enzymexlamlir-opt --llvm-to-affine-access --split-input-file %s | FileCheck %s

// An affine floordiv divides by a positive value -- it is what its lowering and
// the flattening the maps go through are written for, and a negative divisor
// gets as far as "division by non-positive value is not supported". One reaches
// here all the same: instcombine says `-x / c` as `x / -c`. It is the same
// division with the sign taken out, and the expression can say that instead.

llvm.func @neg_div(%p: !llvm.ptr) {
  %cm2 = arith.constant -2 : i64
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 16 {
    %iv = arith.index_cast %i : index to i64
    %d = arith.divsi %iv, %cm2 : i64
    %g = llvm.getelementptr inbounds %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @neg_div
// CHECK:         affine.store %{{.*}}[-(%{{.*}} floordiv 2)]

// -----

// Dividing by zero is not a division this can say at all.

llvm.func @zero_div(%p: !llvm.ptr) {
  %c0 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 16 {
    %iv = arith.index_cast %i : index to i64
    %d = arith.divsi %iv, %c0 : i64
    %g = llvm.getelementptr inbounds %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @zero_div
// CHECK:         %[[D:.+]] = arith.divsi
// CHECK:         llvm.getelementptr inbounds %arg0[%[[D]]]
// CHECK:         memref.store

// -----

// Neither side of this product is free of the loop variable, so it is not a
// product an affine expression has.

llvm.func @dim_times_dim(%p: !llvm.ptr) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %a = arith.index_cast %i : index to i64
      %b = arith.index_cast %j : index to i64
      %m = arith.muli %a, %b : i64
      %g = llvm.getelementptr inbounds %p[%m] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %cst, %g : f64, !llvm.ptr
    }
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @dim_times_dim
// CHECK:         %[[M:.+]] = arith.muli
// CHECK:         llvm.getelementptr inbounds %arg0[%[[M]]]
// CHECK:         memref.store

// -----

// The power of two a shift stands for is worked out at the width the expression
// holds. Taken as an int, `1 << 31` is the largest negative number there is,
// and a shift by 31 is how a sign bit is read.

llvm.func @shr_31(%p: !llvm.ptr, %n: i64) {
  %c31 = arith.constant 31 : i64
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 16 {
    %iv = arith.index_cast %i : index to i64
    %d = arith.shrui %iv, %c31 : i64
    %g = llvm.getelementptr inbounds %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @shr_31
// CHECK:         affine.store %{{.*}}[%{{.*}} floordiv 2147483648]

// -----

// Past 62 there is no power of two a signed affine constant can hold.

llvm.func @shr_63(%p: !llvm.ptr) {
  %c63 = arith.constant 63 : i64
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 16 {
    %iv = arith.index_cast %i : index to i64
    %d = arith.shrui %iv, %c63 : i64
    %g = llvm.getelementptr inbounds %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %cst, %g : f64, !llvm.ptr
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @shr_63
// CHECK:         %[[S:.+]] = arith.shrui
// CHECK:         llvm.getelementptr inbounds %arg0[%[[S]]]
// CHECK:         memref.store
