// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

#set = affine_set<(d0): (-d0 + 99 >= 0)> // d0 <= 99
module {
  func.func @main(%arg0: memref<100x130xf32>, %dim0: index) {
    %val = arith.constant 42.0 : f32
    affine.parallel (%i, %j) = (0, 0) to (120, symbol(%dim0)) {

      %i_i64 = arith.index_cast %i : index to i64
      %i_f32 = arith.sitofp %i_i64 : i64 to f32
      %new_val = arith.mulf %val, %i_f32 : f32

      affine.if #set(%i) {
        affine.store %new_val, %arg0[%i, %j] : memref<100x130xf32>
        affine.yield
      }
    }
    return
  }

  func.func @main2(%arg0: memref<100x130xf32>, %dim0: index) {
    %val = arith.constant 42.0 : f32
    affine.for %k = 0 to %dim0 step 1 {
        affine.parallel (%i, %j) = (0, 0) to (120, %k) {

          %i_i64 = arith.index_cast %i : index to i64
          %i_f32 = arith.sitofp %i_i64 : i64 to f32
          %new_val = arith.mulf %val, %i_f32 : f32

          affine.if #set(%i) {
            affine.store %new_val, %arg0[%i, %j] : memref<100x130xf32>
            affine.yield
          }
        }
    }
    return
  }

}

// CHECK:  func.func @main(%arg0: memref<100x130xf32>, %arg1: index) {
// CHECK-NEXT:    %cst = arith.constant 4.200000e+01 : f32
// CHECK-NEXT:    affine.parallel (%arg2, %arg3) = (0, 0) to (100, symbol(%arg1)) {
// CHECK-NEXT:      %0 = arith.index_cast %arg2 : index to i64
// CHECK-NEXT:      %1 = arith.sitofp %0 : i64 to f32
// CHECK-NEXT:      %2 = arith.mulf %1, %cst : f32
// CHECK-NEXT:      affine.store %2, %arg0[%arg2, %arg3] : memref<100x130xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: memref<100x130xf32>, %arg1: index) {
// CHECK-NEXT:    %cst = arith.constant 4.200000e+01 : f32
// CHECK-NEXT:    affine.for %arg2 = 0 to %arg1 {
// CHECK-NEXT:      affine.parallel (%arg3, %arg4) = (0, 0) to (100, %arg2) {
// CHECK-NEXT:        %0 = arith.index_cast %arg3 : index to i64
// CHECK-NEXT:        %1 = arith.sitofp %0 : i64 to f32
// CHECK-NEXT:        %2 = arith.mulf %1, %cst : f32
// CHECK-NEXT:        affine.store %2, %arg0[%arg3, %arg4] : memref<100x130xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
