// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,arith-raise,enzyme-hlo-opt{max_constant_expansion=1})" | FileCheck %s

// Julia 1.13's LLVM emits `uitofp nneg` for Float64(x) of a provably
// non-negative Int; raised to tensors it must export to XLA.

module {
  func.func private @nneg(%arg0: memref<4x6xf64, 1>, %arg1: memref<4x6xi64, 1>) {
    affine.parallel (%i, %j) = (0, 0) to (4, 6) {
      %v = affine.load %arg1[%i, %j] : memref<4x6xi64, 1>
      %c = arith.uitofp %v nneg : i64 to f64
      affine.store %c, %arg0[%i, %j] : memref<4x6xf64, 1>
    }
    return
  }
  func.func private @plain(%arg0: memref<4x6xf64, 1>, %arg1: memref<4x6xi64, 1>) {
    affine.parallel (%i, %j) = (0, 0) to (4, 6) {
      %v = affine.load %arg1[%i, %j] : memref<4x6xi64, 1>
      %c = arith.uitofp %v : i64 to f64
      affine.store %c, %arg0[%i, %j] : memref<4x6xf64, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @plain_raised(%arg0: tensor<4x6xf64>, %arg1: tensor<4x6xi64>) -> (tensor<4x6xf64>, tensor<4x6xi64>) {
// CHECK-NEXT:    %0 = stablehlo.bitcast_convert %arg1 : (tensor<4x6xi64>) -> tensor<4x6xui64>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : (tensor<4x6xui64>) -> tensor<4x6xf64>
// CHECK-NEXT:    return %1, %arg1 : tensor<4x6xf64>, tensor<4x6xi64>

// CHECK-LABEL: func.func private @nneg_raised(%arg0: tensor<4x6xf64>, %arg1: tensor<4x6xi64>) -> (tensor<4x6xf64>, tensor<4x6xi64>) {
// CHECK-NEXT:    %0 = stablehlo.convert %arg1 : (tensor<4x6xi64>) -> tensor<4x6xf64>
// CHECK-NEXT:    return %0, %arg1 : tensor<4x6xf64>, tensor<4x6xi64>
