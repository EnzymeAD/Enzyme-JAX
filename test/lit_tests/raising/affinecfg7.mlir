// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

 module {
   func.func private @main(%arg0: memref<18x46x46xf64, 1>) {
    %c46 = arith.constant 46 : index
    %c2_i64 = arith.constant 2 : i64
    %c34_i64 = arith.constant 34 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (2, 4, 16) {
      %3 = arith.index_cast %c34_i64 : i64 to index
      %4 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 16)>(%arg2, %arg3, %arg1)
      %5 = arith.addi %4, %3 : index
      %6 = arith.remui %5, %c46 : index
      %7 = arith.divui %5, %c46 : index
      %8 = arith.remui %7, %c46 : index
      %9 = arith.divui %7, %c46 : index
      %10 = memref.load %arg0[%9, %8, %6] : memref<18x46x46xf64, 1>
      affine.store %10, %arg0[%arg2, 39, %arg3 + %arg1 * 16 + 7] : memref<18x46x46xf64, 1>
    }
    return
  }
}


// CHECK:  func.func private @main(%[[MEMREF:.+]]: memref<18x46x46xf64, 1>) {
// CHECK:    affine.parallel (%[[IV:.+]], %[[IV2:.+]]) = (0, 0) to (4, 32) {
// CHECK:    affine.store %[[VAL:.+]], %arg0[%[[IV]], 39, %[[IV2]] + 7] : memref<18x46x46xf64, 1>
