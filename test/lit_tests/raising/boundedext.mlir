// RUN: enzymexlamlir-opt %s --canonicalize-loops --split-input-file | FileCheck %s

func.func private @bar(i64)

func.func private @extcast() {
    affine.parallel (%arg2) = (0) to (256) {
      %0 = arith.index_castui %arg2 : index to i32
      %1 = arith.extui %0 : i32 to i64
      func.call @bar(%1) : (i64) -> ()
    }
    return
}

// CHECK:  func.func private @extcast() {
// CHECK-NEXT:    affine.parallel (%arg0) = (0) to (256) {
// CHECK-NEXT:      %0 = arith.index_castui %arg0 : index to i64
// CHECK-NEXT:      func.call @bar(%0) : (i64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func private @bigger(%arg0: memref<1x134x374xf64, 1>) {
    %c1_i32 = arith.constant 1 : i32
    %c-2_i64 = arith.constant -2 : i64
    %c8_i32 = arith.constant 8 : i32
    %c8_i64 = arith.constant 8 : i64
    %c1_i64 = arith.constant 1 : i64
    %c101_i64 = arith.constant 101 : i64
    %c-361_i64 = arith.constant -361 : i64
    %c-360_i64 = arith.constant -360 : i64
    affine.parallel (%arg1, %arg2) = (0, 0) to (2, 256) {
      %0 = arith.index_castui %arg2 : index to i32
      %1 = arith.extui %0 : i32 to i64
      %2 = arith.index_castui %arg1 : index to i32
      %3 = arith.extui %2 : i32 to i64
      %4 = arith.shrui %2, %c1_i32 : i32
      %5 = arith.extui %4 : i32 to i64
      %6 = arith.muli %5, %c-2_i64 : i64
      %7 = arith.addi %6, %3 : i64
      %8 = arith.shrui %0, %c8_i32 : i32
      %9 = arith.extui %8 : i32 to i64
      %10 = arith.subi %7, %9 : i64
      %11 = arith.shli %10, %c8_i64 : i64
      %12 = arith.addi %1, %c1_i64 : i64
      %13 = arith.addi %12, %11 : i64
      %14 = arith.addi %9, %c101_i64 : i64
      %15 = arith.addi %14, %5 : i64
      %16 = arith.addi %13, %c-361_i64 : i64
      %17 = arith.cmpi ult, %16, %c-360_i64 : i64
      %18 = arith.cmpi ne, %15, %c101_i64 : i64
      %19 = arith.ori %18, %17 : i1
      scf.if %19 {
      } else {
        %20 = affine.load %arg0[((%arg2 + %arg1 * 256 + 2625) floordiv 374) floordiv 134, ((%arg2 + %arg1 * 256 + 2625) floordiv 374) mod 134, %arg2 + %arg1 * 256 - ((%arg2 + %arg1 * 256 + 2625) floordiv 374) * 374 + 2625] : memref<1x134x374xf64, 1>
        affine.store %20, %arg0[((%arg2 + %arg1 * 256 + 2251) floordiv 374) floordiv 134, ((%arg2 + %arg1 * 256 + 2251) floordiv 374) mod 134, %arg2 + %arg1 * 256 - ((%arg2 + %arg1 * 256 + 2251) floordiv 374) * 374 + 2251] : memref<1x134x374xf64, 1>
        %21 = affine.load %arg0[((%arg2 + %arg1 * 256 + 47131) floordiv 374) floordiv 134, ((%arg2 + %arg1 * 256 + 47131) floordiv 374) mod 134, %arg2 + %arg1 * 256 - ((%arg2 + %arg1 * 256 + 47131) floordiv 374) * 374 + 47131] : memref<1x134x374xf64, 1>
        affine.store %21, %arg0[((%arg2 + %arg1 * 256 + 47505) floordiv 374) floordiv 134, ((%arg2 + %arg1 * 256 + 47505) floordiv 374) mod 134, %arg2 + %arg1 * 256 - ((%arg2 + %arg1 * 256 + 47505) floordiv 374) * 374 + 47505] : memref<1x134x374xf64, 1>
      }
    }
    return
  }

// CHECK: @bigger
// CHECK-NOT: index to i32


