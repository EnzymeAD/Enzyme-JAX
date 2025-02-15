// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(func.func(canonicalize-loops), canonicalize)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: foo
// CHECK-SAME: %[[ARG0:.+]]: memref<256xi64>
func.func private @foo(%arg0: memref<256xi64>) {
  %c134_i64 = arith.constant 134 : i64
  %c1_i32 = arith.constant 1 : i32
  // CHECK: %[[C0:.+]] = arith.constant 0 : i64
  // CHECK-NEXT: affine.parallel (%[[ARG1:.+]]) = (0) to (133)
  affine.parallel (%arg1) = (0) to (133) {
    %0 = arith.index_cast %arg1 : index to i32
    %1 = arith.addi %0, %c1_i32 : i32
    %2 = arith.extui %1 : i32 to i64
    %3 = arith.divui %2, %c134_i64 : i64
    %4 = arith.addi %3, %3 : i64
    // CHECK: affine.store %[[C0]], %[[ARG0]][%[[ARG1]]] : memref<256xi64>
    affine.store %4, %arg0[%arg1] : memref<256xi64>
  }
  return
}

// -----

func.func private @foo(%arg0: memref<256xi64>) {
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c3_i64 = arith.constant 3 : i64
  %c134_i16 = arith.constant 134 : i16
  affine.parallel (%arg1) = (0) to (134) {
    %0 = arith.index_cast %arg1 : index to i32
    %2 = arith.trunci %0 : i32 to i16
    %3 = arith.divui %2, %c134_i16 : i16
    %4 = arith.extui %3 : i16 to i64
    %8 = arith.addi %4, %c1_i64 : i64
    // This condition is always false since the loop-bound range is [0, 134).
    // If we trach the use-def-chain, we notice that the loop induction
    // variable is first integer-divided by 134 and then incremented by one.
    // So i/134 is 0, 0 +1 is 1, and 1 != 1 is false.
    %11 = arith.cmpi ne, %8, %c1_i64 : i64
    scf.if %11 {
      affine.store %c2_i64, %arg0[%arg1] : memref<256xi64>
    } else {
      affine.store %c3_i64, %arg0[%arg1] : memref<256xi64>
    }
  }
  return
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:.+]]: memref<256xi64>
// CHECK: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-NEXT: affine.parallel (%[[ARG1:.+]]) = (0) to (134)
// CHECK: affine.store %[[C3]], %[[ARG0]][%[[ARG1]]] : memref<256xi64>

// -----

func.func private @foo(%arg0: memref<256xi64>) {
  %c2_i64 = arith.constant 2 : i64
  %c3_i64 = arith.constant 3 : i64
  %c-134_i64 = arith.constant -134 : i64
  affine.parallel (%arg1) = (0) to (134) {
    %0 = arith.index_cast %arg1 : index to i32
    %1 = arith.extui %0 : i32 to i64
    %2 = arith.addi %1, %c-134_i64 : i64
    %3 = arith.cmpi ult, %2, %c-134_i64 : i64
    scf.if %3 {
      affine.store %c2_i64, %arg0[%arg1] : memref<256xi64>
    } else {
      affine.store %c3_i64, %arg0[%arg1] : memref<256xi64>
    }
  }
  return
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:.+]]: memref<256xi64>
// CHECK: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-NEXT: affine.parallel (%[[ARG1:.+]]) = (0) to (134)
// CHECK: affine.store %[[C3]], %[[ARG0]][%[[ARG1]]] : memref<256xi64>

// -----

func.func private @foo(%arg0: memref<1x134x374xf64, 1>) {
  %c1_i32 = arith.constant 1 : i32
  %c134_i16 = arith.constant 134 : i16
  %c-134_i64 = arith.constant -134 : i64
  %c1_i64 = arith.constant 1 : i64
  %c-135_i64 = arith.constant -135 : i64
  affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (1, 1, 1, 134, 1, 1) {
    %0 = arith.index_cast %arg1 : index to i32
    %1 = arith.addi %0, %c1_i32 : i32
    %2 = arith.extui %1 : i32 to i64
    %3 = arith.index_cast %arg4 : index to i32
    %4 = arith.extui %3 : i32 to i64
    %5 = arith.trunci %3 : i32 to i16
    %6 = arith.divui %5, %c134_i16 : i16
    %7 = arith.extui %6 : i16 to i64
    %8 = arith.muli %7, %c-134_i64 : i64
    %9 = arith.addi %4, %c1_i64 : i64
    %10 = arith.addi %9, %8 : i64
    %11 = arith.addi %2, %7 : i64
    %12 = arith.addi %10, %c-135_i64 : i64
    %13 = arith.cmpi ult, %12, %c-134_i64 : i64
    %14 = arith.cmpi ne, %11, %c1_i64 : i64
    %15 = arith.ori %14, %13 : i1
    scf.if %15 {
    } else {
      %16 = affine.load %arg0[0, %arg4, 360] : memref<1x134x374xf64, 1>
      affine.store %16, %arg0[0, %arg4, 0] : memref<1x134x374xf64, 1>
      %17 = affine.load %arg0[0, %arg4, 7] : memref<1x134x374xf64, 1>
      affine.store %17, %arg0[0, %arg4, 367] : memref<1x134x374xf64, 1>
      %18 = affine.load %arg0[0, %arg4, 361] : memref<1x134x374xf64, 1>
      affine.store %18, %arg0[0, %arg4, 1] : memref<1x134x374xf64, 1>
      %19 = affine.load %arg0[0, %arg4, 8] : memref<1x134x374xf64, 1>
      affine.store %19, %arg0[0, %arg4, 368] : memref<1x134x374xf64, 1>
      %20 = affine.load %arg0[0, %arg4, 362] : memref<1x134x374xf64, 1>
      affine.store %20, %arg0[0, %arg4, 2] : memref<1x134x374xf64, 1>
      %21 = affine.load %arg0[0, %arg4, 9] : memref<1x134x374xf64, 1>
      affine.store %21, %arg0[0, %arg4, 369] : memref<1x134x374xf64, 1>
      %22 = affine.load %arg0[0, %arg4, 363] : memref<1x134x374xf64, 1>
      affine.store %22, %arg0[0, %arg4, 3] : memref<1x134x374xf64, 1>
      %23 = affine.load %arg0[0, %arg4, 10] : memref<1x134x374xf64, 1>
      affine.store %23, %arg0[0, %arg4, 370] : memref<1x134x374xf64, 1>
      %24 = affine.load %arg0[0, %arg4, 364] : memref<1x134x374xf64, 1>
      affine.store %24, %arg0[0, %arg4, 4] : memref<1x134x374xf64, 1>
      %25 = affine.load %arg0[0, %arg4, 11] : memref<1x134x374xf64, 1>
      affine.store %25, %arg0[0, %arg4, 371] : memref<1x134x374xf64, 1>
      %26 = affine.load %arg0[0, %arg4, 365] : memref<1x134x374xf64, 1>
      affine.store %26, %arg0[0, %arg4, 5] : memref<1x134x374xf64, 1>
      %27 = affine.load %arg0[0, %arg4, 12] : memref<1x134x374xf64, 1>
      affine.store %27, %arg0[0, %arg4, 372] : memref<1x134x374xf64, 1>
      %28 = affine.load %arg0[0, %arg4, 366] : memref<1x134x374xf64, 1>
      affine.store %28, %arg0[0, %arg4, 6] : memref<1x134x374xf64, 1>
      %29 = affine.load %arg0[0, %arg4, 13] : memref<1x134x374xf64, 1>
      affine.store %29, %arg0[0, %arg4, 373] : memref<1x134x374xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:.+]]: memref<1x134x374xf64, 1>
// CHECK: affine.parallel (%[[ARG1:.+]]) = (0) to (134) {
// CHECK-NEXT: %[[V0:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 360] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V0]], %[[ARG0]][0, %[[ARG1]], 0] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V1:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 7] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V1]], %[[ARG0]][0, %[[ARG1]], 367] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V2:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 361] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V2]], %[[ARG0]][0, %[[ARG1]], 1] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V3:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 8] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V3]], %[[ARG0]][0, %[[ARG1]], 368] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V4:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 362] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V4]], %[[ARG0]][0, %[[ARG1]], 2] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V5:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 9] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V5]], %[[ARG0]][0, %[[ARG1]], 369] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V6:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 363] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V6]], %[[ARG0]][0, %[[ARG1]], 3] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V7:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 10] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V7]], %[[ARG0]][0, %[[ARG1]], 370] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V8:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 364] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V8]], %[[ARG0]][0, %[[ARG1]], 4] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V9:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 11] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V9]], %[[ARG0]][0, %[[ARG1]], 371] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V10:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 365] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V10]], %[[ARG0]][0, %[[ARG1]], 5] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V11:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 12] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V11]], %[[ARG0]][0, %[[ARG1]], 372] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V12:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 366] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V12]], %[[ARG0]][0, %[[ARG1]], 6] : memref<1x134x374xf64, 1>
// CHECK-NEXT: %[[V13:.+]] = affine.load %[[ARG0]][0, %[[ARG1]], 13] : memref<1x134x374xf64, 1>
// CHECK-NEXT: affine.store %[[V13]], %[[ARG0]][0, %[[ARG1]], 373] : memref<1x134x374xf64, 1>
// CHECK-NEXT: }
// CHECK-NEXT: return