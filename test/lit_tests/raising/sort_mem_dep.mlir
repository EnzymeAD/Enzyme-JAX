// RUN: enzymexlamlir-opt %s --sort-memory | FileCheck %s

module @reactant_bad_ker... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @split_explicit_barotropic_velocity(%arg0: memref<1x112x208xf64, 1>, %arg2: memref<1x112x208xf64, 1>) {
    %cst = arith.constant -39222.659999999996 : f64
    affine.parallel (%arg4, %arg5) = (0, 0) to (96, 192) {
      %0 = affine.load %arg0[0, %arg4 + 8, %arg5 + 8] : memref<1x112x208xf64, 1>
      %1 = arith.addf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %1, %arg0[0, %arg4 + 8, %arg5 + 8] : memref<1x112x208xf64, 1>
      %4 = affine.load %arg2[0, %arg4 + 8, %arg5 + 8] : memref<1x112x208xf64, 1>
      %5 = affine.load %arg0[0, %arg4 + 8, %arg5 + 8] : memref<1x112x208xf64, 1>
      %6 = arith.addf %4, %5 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %6, %arg2[0, %arg4 + 8, %arg5 + 8] : memref<1x112x208xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @split_explicit_barotropic_velocity(%arg0: memref<1x112x208xf64, 1>, %arg1: memref<1x112x208xf64, 1>) {
// CHECK-NEXT:    %cst = arith.constant -39222.659999999996 : f64
// CHECK-NEXT:    affine.parallel (%arg2, %arg3) = (0, 0) to (96, 192) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, %arg2 + 8, %arg3 + 8] : memref<1x112x208xf64, 1>
// CHECK-NEXT:      %1 = arith.addf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %1, %arg0[0, %arg2 + 8, %arg3 + 8] : memref<1x112x208xf64, 1>
// CHECK-NEXT:      %2 = affine.load %arg1[0, %arg2 + 8, %arg3 + 8] : memref<1x112x208xf64, 1>
// CHECK-NEXT:      %3 = affine.load %arg0[0, %arg2 + 8, %arg3 + 8] : memref<1x112x208xf64, 1>
// CHECK-NEXT:      %4 = arith.addf %2, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %4, %arg1[0, %arg2 + 8, %arg3 + 8] : memref<1x112x208xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
