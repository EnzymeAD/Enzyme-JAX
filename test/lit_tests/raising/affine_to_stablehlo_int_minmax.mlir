// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-stablehlo --canonicalize | FileCheck %s

// The longest row of a CSR structure, a signed max over its offsets.
func.func private @row_max(%off: memref<101xi32, 1>, %out: memref<i32, 1>) {
  %0 = affine.parallel (%i) = (0) to (100) reduce ("maxs") -> (i32) {
    %lo = affine.load %off[%i] : memref<101xi32, 1>
    %hi = affine.load %off[%i + 1] : memref<101xi32, 1>
    %d = arith.subi %hi, %lo : i32
    affine.yield %d : i32
  }
  affine.store %0, %out[] : memref<i32, 1>
  return
}

// CHECK:  func.func private @row_max_raised(%arg0: tensor<101xi32>, %arg1: tensor<i32>) -> (tensor<101xi32>, tensor<i32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<-2147483648> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:100] : (tensor<101xi32>) -> tensor<100xi32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [1:101] : (tensor<101xi32>) -> tensor<100xi32>
// CHECK-NEXT:    %2 = arith.subi %1, %0 : tensor<100xi32>
// CHECK-NEXT:    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.maximum across dimensions = [0] : (tensor<100xi32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT:    %4 = stablehlo.dynamic_update_slice %arg1, %3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT:    return %arg0, %4 : tensor<101xi32>, tensor<i32>
// CHECK-NEXT:  }

// -----

func.func private @col_min(%in: memref<5x20xi64, 1>, %out: memref<20xi64, 1>) {
  affine.parallel (%j) = (0) to (20) {
    %0 = affine.parallel (%i) = (0) to (5) reduce ("mins") -> (i64) {
      %v = affine.load %in[%i, %j] : memref<5x20xi64, 1>
      affine.yield %v : i64
    }
    affine.store %0, %out[%j] : memref<20xi64, 1>
  }
  return
}

// CHECK:  func.func private @col_min_raised(%arg0: tensor<5x20xi64>, %arg1: tensor<20xi64>) -> (tensor<5x20xi64>, tensor<20xi64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<9223372036854775807> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.minimum across dimensions = [0] : (tensor<5x20xi64>, tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg1, %0, %c : (tensor<20xi64>, tensor<20xi64>, tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    return %arg0, %1 : tensor<5x20xi64>, tensor<20xi64>
// CHECK-NEXT:  }

// -----

func.func private @col_minmax_unsigned(%in: memref<5x20xi32, 1>, %out: memref<2x20xi32, 1>) {
  affine.parallel (%j) = (0) to (20) {
    %0 = affine.parallel (%i) = (0) to (5) reduce ("maxu") -> (i32) {
      %v = affine.load %in[%i, %j] : memref<5x20xi32, 1>
      affine.yield %v : i32
    }
    %1 = affine.parallel (%i) = (0) to (5) reduce ("minu") -> (i32) {
      %v = affine.load %in[%i, %j] : memref<5x20xi32, 1>
      affine.yield %v : i32
    }
    affine.store %0, %out[0, %j] : memref<2x20xi32, 1>
    affine.store %1, %out[1, %j] : memref<2x20xi32, 1>
  }
  return
}

// CHECK:  func.func private @col_minmax_unsigned_raised(%arg0: tensor<5x20xi32>, %arg1: tensor<2x20xi32>) -> (tensor<5x20xi32>, tensor<2x20xi32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<-1> : tensor<i32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %c_2) across dimensions = [0] : (tensor<5x20xi32>, tensor<i32>) -> tensor<20xi32>
// CHECK-NEXT:     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
// CHECK-NEXT:      %7 = arith.maxui %arg2, %arg3 : tensor<i32>
// CHECK-NEXT:      stablehlo.return %7 : tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<5x20xi32>) -> tensor<5x20xi32>
// CHECK-NEXT:    %2 = stablehlo.reduce(%1 init: %c_1) across dimensions = [0] : (tensor<5x20xi32>, tensor<i32>) -> tensor<20xi32>
// CHECK-NEXT:     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
// CHECK-NEXT:      %7 = arith.minui %arg2, %arg3 : tensor<i32>
// CHECK-NEXT:      stablehlo.return %7 : tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<20xi32>) -> tensor<1x20xi32>
// CHECK-NEXT:    %4 = stablehlo.dynamic_update_slice %arg1, %3, %c_0, %c_0 : (tensor<2x20xi32>, tensor<1x20xi32>, tensor<i64>, tensor<i64>) -> tensor<2x20xi32>
// CHECK-NEXT:    %5 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<20xi32>) -> tensor<1x20xi32>
// CHECK-NEXT:    %6 = stablehlo.dynamic_update_slice %4, %5, %c, %c_0 : (tensor<2x20xi32>, tensor<1x20xi32>, tensor<i64>, tensor<i64>) -> tensor<2x20xi32>
// CHECK-NEXT:    return %arg0, %6 : tensor<5x20xi32>, tensor<2x20xi32>
// CHECK-NEXT:  }
