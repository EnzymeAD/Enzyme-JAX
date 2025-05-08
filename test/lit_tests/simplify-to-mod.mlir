// RUN: enzymexlamlir-opt %s --simplify-affine-exprs --split-input-file | FileCheck %s

#set6 = affine_set<(d0, d1) : (-(d0 floordiv 16) - (d1 floordiv 7) * 16 + 33 >= 0, d0 + d1 * 16 - (d1 floordiv 7) * 112 - (d0 floordiv 16) * 16 >= 0, -d0 - d1 * 16 + (d1 floordiv 7) * 112 + (d0 floordiv 16) * 16 + 98 >= 0)>

// CHECK: #set = affine_set<(d0, d1) : (-(d0 floordiv 16) - (d1 floordiv 7) * 16 + 33 >= 0, d0 mod 16 + (d1 mod 7) * 16 >= 0, -(d0 mod 16) - (d1 mod 7) * 16 + 98 >= 0)>

module {

  func.func private @kern1(%memref_arg: memref<20x30xi64, 1>) {
    affine.parallel (%arg2) = (0) to (21) {
      %2 = arith.constant 1 : i64
      affine.store %2, %memref_arg[%arg2 floordiv 7, %arg2 - (%arg2 floordiv 7) * 7] : memref<20x30xi64, 1>
    }
    return
  }

  // CHECK-LABEL: kern1
  // CHECK:  affine.store %c1_i64, %arg0[%arg1 floordiv 7, %arg1 mod 7]

  func.func private @kern2(%arg0: memref<34x99xf64, 1>) {
    affine.parallel (%arg5, %arg6) = (0, 0) to (21, 256) {
      affine.if #set6(%arg6, %arg5) {
        %cst = arith.constant 1.0 : f64
        affine.store %cst, %arg0[(%arg6 + %arg5 * 16 + (%arg6 floordiv 16) * 83 + (%arg5 floordiv 7) * 1472) floordiv 99, (%arg6 + %arg5 * 16 + (%arg6 floordiv 16) * 83 + (%arg5 floordiv 7) * 1472)mod 99] : memref<34x99xf64, 1>
      }
    }
    return
  }
  
  // CHECK-LABEL: kern2
  // CHECK:  affine.store %cst, %arg0[%arg2 floordiv 16 + (%arg1 floordiv 7) * 16, %arg2 mod 16 + (%arg1 mod 7) * 16]

  func.func private @kern3(%arg0: memref<99xf64, 1>) {
    affine.parallel (%arg1, %arg2) = (0, 0) to (21, 256) {
      %cst = arith.constant 1.000000e+00 : f64
      affine.store %cst, %arg0[1 + %arg2 - (%arg2 floordiv 16) * 16] : memref<99xf64, 1>
      affine.store %cst, %arg0[%arg1 + %arg2 - (%arg2 floordiv 16) * 16] : memref<99xf64, 1>
      affine.store %cst, %arg0[%arg1 * 16 + %arg2 - (%arg1 floordiv 7) * 112 - (%arg2 floordiv 16) * 16] : memref<99xf64, 1>
    }
    return
  }
  
  // CHECK-LABEL: kern3
  // CHECK:    affine.store %cst, %arg0[%arg2 mod 16 + 1] : memref<99xf64, 1>
  // CHECK:    affine.store %cst, %arg0[%arg1 + %arg2 mod 16] : memref<99xf64, 1>
  // CHECK:    affine.store %cst, %arg0[%arg2 mod 16 + (%arg1 mod 7) * 16] : memref<99xf64, 1>
}
