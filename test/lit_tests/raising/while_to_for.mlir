// RUN: enzymexlamlir-opt %s  --canonicalize-scf-for --split-input-file | FileCheck %s

// Check that we correctly identify ambigous main IV

func.func @foo(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<35xf64, 1>, %arg2: memref<34xf64, 1>) {
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c1_i64 = arith.constant 1 : i64
  %c20_i64 = arith.constant 20 : i64
  affine.parallel (%arg3, %arg4) = (0, 0) to (90, 180) {
    %0 = affine.load %arg0[0, %arg3 + 7, %arg4 + 7] : memref<1x104x194xf64, 1>
    %1 = affine.load %arg1[7] : memref<35xf64, 1>
    affine.store %1, %arg0[0, %arg3 + 7, %arg4 + 7] : memref<1x104x194xf64, 1>
    %2:2 = scf.while (%arg5 = %1, %arg6 = %c1_i64) : (f64, i64) -> (f64, i64) {
      %3 = arith.index_cast %arg6 : i64 to index
      %4 = arith.addi %3, %c7 : index
      %5 = memref.load %arg1[%4] : memref<35xf64, 1>
      %6 = arith.index_cast %arg6 : i64 to index
      %7 = arith.addi %6, %c6 : index
      %8 = memref.load %arg2[%7] : memref<34xf64, 1>
      %9 = arith.cmpf ole, %8, %0 : f64
      %10 = arith.select %9, %5, %arg5 : f64
      affine.store %10, %arg0[0, %arg3 + 7, %arg4 + 7] : memref<1x104x194xf64, 1>
      %11 = arith.addi %arg6, %c1_i64 : i64
      %12 = arith.cmpi ne, %arg6, %c20_i64 : i64
      scf.condition(%12) %10, %11 : f64, i64
    } do {
    ^bb0(%arg5: f64, %arg6: i64):
      scf.yield %arg5, %arg6 : f64, i64
    }
  }
  return
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<1x104x194xf64, 1>,
// CHECK-SAME:                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<35xf64, 1>,
// CHECK-SAME:                   %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<34xf64, 1>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 21 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 7 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           affine.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (0, 0) to (90, 180) {
// CHECK:             %[[VAL_9:.*]] = affine.load %[[VAL_0]][0, %[[VAL_7]] + 7, %[[VAL_8]] + 7] : memref<1x104x194xf64, 1>
// CHECK:             %[[VAL_10:.*]] = affine.load %[[VAL_1]][7] : memref<35xf64, 1>
// CHECK:             affine.store %[[VAL_10]], %[[VAL_0]][0, %[[VAL_7]] + 7, %[[VAL_8]] + 7] : memref<1x104x194xf64, 1>
// CHECK:             %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_6]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (f64)  : i64 {
// CHECK:               %[[VAL_14:.*]] = arith.index_cast %[[VAL_12]] : i64 to index
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_5]] : index
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_15]]] : memref<35xf64, 1>
// CHECK:               %[[VAL_17:.*]] = arith.index_cast %[[VAL_12]] : i64 to index
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_4]] : index
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_18]]] : memref<34xf64, 1>
// CHECK:               %[[VAL_20:.*]] = arith.cmpf ole, %[[VAL_19]], %[[VAL_9]] : f64
// CHECK:               %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_16]], %[[VAL_13]] : f64
// CHECK:               affine.store %[[VAL_21]], %[[VAL_0]][0, %[[VAL_7]] + 7, %[[VAL_8]] + 7] : memref<1x104x194xf64, 1>
// CHECK:               scf.yield %[[VAL_21]] : f64
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

