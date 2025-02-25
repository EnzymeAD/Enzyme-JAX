// RUN: enzymexlamlir-opt %s --simplify-affine-exprs --split-input-file | FileCheck %s

func.func private @kern$par0(%memref_arg: memref<?x20x30xi64, 1>, %idx : index, %idx2 : index) {
  affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 30) to (10, 20, 50) {
    %2 = arith.constant 1 : i64
    affine.store %2, %memref_arg[%arg1 + symbol(%idx), %arg2 floordiv 30, %arg3 mod 30] : memref<?x20x30xi64, 1>
    affine.store %2, %memref_arg[%arg1, %arg2 floordiv 30, %arg3 mod 30] : memref<?x20x30xi64, 1>
    %l = affine.load %memref_arg[%arg1, %arg2 floordiv 10, %arg3 mod 20] : memref<?x20x30xi64, 1>
    %l2 = affine.load %memref_arg[%arg1 + symbol(%idx2), %arg2 floordiv 10, %arg3 mod 20] : memref<?x20x30xi64, 1>
  }
  return
}


// CHECK-LABEL:   func.func private @kern$par0(
// CHECK-SAME:                                 %[[VAL_0:[^:]*]]: memref<?x20x30xi64, 1>,
// CHECK-SAME:                                 %[[VAL_1:[^:]*]]: index,
// CHECK-SAME:                                 %[[VAL_2:[^:]*]]: index) {
// CHECK:           affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) = (0, 0, 30) to (10, 20, 50) {
// CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:             affine.store %[[VAL_6]], %[[VAL_0]]{{\[}}%[[VAL_3]] + symbol(%[[VAL_1]]), %[[VAL_4]] floordiv 30, %[[VAL_5]] mod 30] : memref<?x20x30xi64, 1>
// CHECK:             affine.store %[[VAL_6]], %[[VAL_0]]{{\[}}%[[VAL_3]], 0, %[[VAL_5]] - 30] : memref<?x20x30xi64, 1>
// CHECK:             %[[VAL_7:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_4]] floordiv 10, %[[VAL_5]] mod 20] : memref<?x20x30xi64, 1>
// CHECK:             %[[VAL_8:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_3]] + symbol(%[[VAL_2]]), %[[VAL_4]] floordiv 10, %[[VAL_5]] mod 20] : memref<?x20x30xi64, 1>
