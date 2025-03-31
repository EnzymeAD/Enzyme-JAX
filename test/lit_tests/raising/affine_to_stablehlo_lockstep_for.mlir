// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func private @positive_dep
// CHECK-NOT: stablehlo.while

module {
  func.func private @positive_dep(%arg0: memref<34x104x194xf64, 1>, %arg1: memref<34x104x194xf64, 1>, %arg2: memref<35x104x194xf64, 1>, %arg3: memref<34xf64, 1>, %arg4: memref<34xf64, 1>, %arg5: memref<104x194xf64, 1>, %arg6: memref<104x194xf64, 1>, %arg7: memref<104x194xf64, 1>, %arg8: memref<1x104x194xf64, 1>) {
    %c-4_i64 = arith.constant -4 : i64
    %c-6_i64 = arith.constant -6 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-5_i64 = arith.constant -5 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f64
    %cst2 = arith.constant 2.000000e+00 : f64
    %c20_i64 = arith.constant 20 : i64
    affine.parallel (%arg9, %arg10) = (0, 0) to (102, 192) {
      %0 = arith.index_castui %arg9 : index to i64
      %1 = arith.addi %0, %c-5_i64 : i64
      affine.store %cst, %arg2[7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
      %2 = arith.cmpi slt, %1, %c1_i64 : i64
      %3 = arith.addi %0, %c-6_i64 : i64
      %4 = arith.cmpi slt, %3, %c1_i64 : i64
      %5 = arith.addi %0, %c-4_i64 : i64
      %6 = arith.cmpi slt, %5, %c1_i64 : i64
      affine.for %arg11 = 0 to 20 {
        %72 = affine.load %arg2[%arg11 + 8, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
        %73 = arith.subf %72, %cst2 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %73, %arg2[%arg11 + 7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
      }
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func private @zero_dep
// CHECK-NOT: stablehlo.while

module {
  func.func private @zero_dep(%arg0: memref<34x104x194xf64, 1>, %arg1: memref<34x104x194xf64, 1>, %arg2: memref<35x104x194xf64, 1>, %arg3: memref<34xf64, 1>, %arg4: memref<34xf64, 1>, %arg5: memref<104x194xf64, 1>, %arg6: memref<104x194xf64, 1>, %arg7: memref<104x194xf64, 1>, %arg8: memref<1x104x194xf64, 1>) {
    %c-4_i64 = arith.constant -4 : i64
    %c-6_i64 = arith.constant -6 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-5_i64 = arith.constant -5 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f64
    %cst2 = arith.constant 2.000000e+00 : f64
    %c20_i64 = arith.constant 20 : i64
    affine.parallel (%arg9, %arg10) = (0, 0) to (102, 192) {
      %0 = arith.index_castui %arg9 : index to i64
      %1 = arith.addi %0, %c-5_i64 : i64
      affine.store %cst, %arg2[7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
      %2 = arith.cmpi slt, %1, %c1_i64 : i64
      %3 = arith.addi %0, %c-6_i64 : i64
      %4 = arith.cmpi slt, %3, %c1_i64 : i64
      %5 = arith.addi %0, %c-4_i64 : i64
      %6 = arith.cmpi slt, %5, %c1_i64 : i64
      affine.for %arg11 = 0 to 20 {
        %72 = affine.load %arg2[%arg11 + 7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
        %73 = arith.subf %72, %cst2 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %73, %arg2[%arg11 + 7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
      }
    }
    return
  }
}

