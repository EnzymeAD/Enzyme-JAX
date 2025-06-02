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

// -----

#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0) : (-d0 + 89 >= 0)>
module {
  // CHECK-LABEL: @par6
  func.func private @par6(%arg0: memref<1x104x194xf64, 1>) {
    %c2_i64 = arith.constant 2 : i64
    %c182_i64 = arith.constant 182 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    affine.parallel (%arg1) = (0) to (180) {
      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x104x194xf64, 1>
      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x104x194xf64, 1>
      %1:10 = affine.if #set(%arg1) -> (i64, i64, f64, f64, f64, f64, f64, f64, f64, f64) {
        // CHECK: affine.load %arg0[0, 96, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 89, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 90, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 91, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 92, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 93, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 94, -%{{.*}} + 187]
        // CHECK: affine.load %arg0[0, 95, -%{{.*}} + 187]
        %13 = affine.load %arg0[((-%arg1 + 18811) floordiv 194) floordiv 104, ((-%arg1 + 18811) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18811) floordiv 194) * 194 + 18811] : memref<1x104x194xf64, 1>
        %14 = affine.load %arg0[((-%arg1 + 17453) floordiv 194) floordiv 104, ((-%arg1 + 17453) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17453) floordiv 194) * 194 + 17453] : memref<1x104x194xf64, 1>
        %15 = affine.load %arg0[((-%arg1 + 17647) floordiv 194) floordiv 104, ((-%arg1 + 17647) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17647) floordiv 194) * 194 + 17647] : memref<1x104x194xf64, 1>
        %16 = affine.load %arg0[((-%arg1 + 17841) floordiv 194) floordiv 104, ((-%arg1 + 17841) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17841) floordiv 194) * 194 + 17841] : memref<1x104x194xf64, 1>
        %17 = affine.load %arg0[((-%arg1 + 18035) floordiv 194) floordiv 104, ((-%arg1 + 18035) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18035) floordiv 194) * 194 + 18035] : memref<1x104x194xf64, 1>
        %18 = affine.load %arg0[((-%arg1 + 18229) floordiv 194) floordiv 104, ((-%arg1 + 18229) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18229) floordiv 194) * 194 + 18229] : memref<1x104x194xf64, 1>
        %19 = affine.load %arg0[((-%arg1 + 18423) floordiv 194) floordiv 104, ((-%arg1 + 18423) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18423) floordiv 194) * 194 + 18423] : memref<1x104x194xf64, 1>
        %20 = affine.load %arg0[((-%arg1 + 18617) floordiv 194) floordiv 104, ((-%arg1 + 18617) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18617) floordiv 194) * 194 + 18617] : memref<1x104x194xf64, 1>
        affine.yield %c-1_i64, %c182_i64, %13, %14, %15, %16, %17, %18, %19, %20 : i64, i64, f64, f64, f64, f64, f64, f64, f64, f64
      } else {
        // CHECK: affine.load %arg0[0, 96, 7]
        // CHECK: affine.load %arg0[0, 89, 7]
        // CHECK: affine.load %arg0[0, 90, 7]
        // CHECK: affine.load %arg0[0, 91, 7]
        // CHECK: affine.load %arg0[0, 92, 7]
        // CHECK: affine.load %arg0[0, 93, 7]
        // CHECK: affine.load %arg0[0, 94, 7]
        // CHECK: affine.load %arg0[0, 95, 7]
        %13 = affine.load %arg0[((-%arg1 + 18631) floordiv 194) floordiv 104, ((-%arg1 + 18631) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18631) floordiv 194) * 194 + 18631] : memref<1x104x194xf64, 1>
        %14 = affine.load %arg0[((-%arg1 + 17273) floordiv 194) floordiv 104, ((-%arg1 + 17273) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17273) floordiv 194) * 194 + 17273] : memref<1x104x194xf64, 1>
        %15 = affine.load %arg0[((-%arg1 + 17467) floordiv 194) floordiv 104, ((-%arg1 + 17467) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17467) floordiv 194) * 194 + 17467] : memref<1x104x194xf64, 1>
        %16 = affine.load %arg0[((-%arg1 + 17661) floordiv 194) floordiv 104, ((-%arg1 + 17661) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17661) floordiv 194) * 194 + 17661] : memref<1x104x194xf64, 1>
        %17 = affine.load %arg0[((-%arg1 + 17855) floordiv 194) floordiv 104, ((-%arg1 + 17855) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 17855) floordiv 194) * 194 + 17855] : memref<1x104x194xf64, 1>
        %18 = affine.load %arg0[((-%arg1 + 18049) floordiv 194) floordiv 104, ((-%arg1 + 18049) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18049) floordiv 194) * 194 + 18049] : memref<1x104x194xf64, 1>
        %19 = affine.load %arg0[((-%arg1 + 18243) floordiv 194) floordiv 104, ((-%arg1 + 18243) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18243) floordiv 194) * 194 + 18243] : memref<1x104x194xf64, 1>
        %20 = affine.load %arg0[((-%arg1 + 18437) floordiv 194) floordiv 104, ((-%arg1 + 18437) floordiv 194) mod 104, -%arg1 - ((-%arg1 + 18437) floordiv 194) * 194 + 18437] : memref<1x104x194xf64, 1>
        affine.yield %c1_i64, %c2_i64, %13, %14, %15, %16, %17, %18, %19, %20 : i64, i64, f64, f64, f64, f64, f64, f64, f64, f64
      }
      %2 = arith.sitofp %1#0 : i64 to f64
      %3 = arith.mulf %2, %1#9 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %3, %arg0[0, 97, %arg1 + 7] : memref<1x104x194xf64, 1>
      %4 = arith.mulf %2, %1#8 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %4, %arg0[0, 98, %arg1 + 7] : memref<1x104x194xf64, 1>
      %5 = arith.mulf %2, %1#7 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %5, %arg0[0, 99, %arg1 + 7] : memref<1x104x194xf64, 1>
      %6 = arith.mulf %2, %1#6 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %6, %arg0[0, 100, %arg1 + 7] : memref<1x104x194xf64, 1>
      %7 = arith.mulf %2, %1#5 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %7, %arg0[0, 101, %arg1 + 7] : memref<1x104x194xf64, 1>
      %8 = arith.mulf %2, %1#4 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %8, %arg0[0, 102, %arg1 + 7] : memref<1x104x194xf64, 1>
      %9 = arith.mulf %2, %1#3 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %9, %arg0[0, 103, %arg1 + 7] : memref<1x104x194xf64, 1>
      %10 = arith.mulf %2, %1#2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %11 = affine.load %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
      %12 = affine.if #set1(%arg1) -> f64 {
        affine.yield %11 : f64
      } else {
        affine.yield %10 : f64
      }
      affine.store %12, %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
    }
    return
  }
}
