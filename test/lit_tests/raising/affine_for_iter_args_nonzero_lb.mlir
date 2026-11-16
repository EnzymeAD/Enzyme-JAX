// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=prefer_while_raising=false --split-input-file | FileCheck %s

// A lockstep-raised reduction loop holds one lane per iteration, position p
// for iv = lb + p * step; the carried value after the loop is the last lane,
// wherever the bounds start (issue #3590: a nonzero lower bound sliced at
// the induction variable's last value instead, past the end of the axis).
func.func private @sum_from_one(%x: memref<8xf64>, %y: memref<f64>) {
  %c0 = arith.constant 0.0 : f64
  %1 = affine.for %k = 1 to 8 iter_args(%acc = %c0) -> (f64) {
    %3 = affine.load %x[%k] : memref<8xf64>
    %4 = arith.addf %acc, %3 : f64
    affine.yield %4 : f64
  }
  affine.store %1, %y[] : memref<f64>
  return
}

// CHECK-LABEL: func.func private @sum_from_one_raised(
// CHECK: %[[X:.+]] = stablehlo.slice %arg0 [1:8] : (tensor<8xf64>) -> tensor<7xf64>
// CHECK: %[[SCAN:.+]] = "stablehlo.reduce_window"(%[[X]], %{{.+}})
// CHECK: %[[ACC:.+]] = stablehlo.add %[[SCAN]], %{{.+}} : tensor<7xf64>
// CHECK: stablehlo.slice %[[ACC]] [6:7] : (tensor<7xf64>) -> tensor<1xf64>

// -----

// The same with a step: four iterations at k = 1, 3, 5, 7.
func.func private @sum_stepped(%x: memref<8xf64>, %y: memref<f64>) {
  %c0 = arith.constant 0.0 : f64
  %1 = affine.for %k = 1 to 8 step 2 iter_args(%acc = %c0) -> (f64) {
    %3 = affine.load %x[%k] : memref<8xf64>
    %4 = arith.addf %acc, %3 : f64
    affine.yield %4 : f64
  }
  affine.store %1, %y[] : memref<f64>
  return
}

// CHECK-LABEL: func.func private @sum_stepped_raised(
// CHECK: %[[X:.+]] = "stablehlo.gather"(%arg0, %{{.+}}) {{.*}} : (tensor<8xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK: %[[SCAN:.+]] = "stablehlo.reduce_window"(%[[X]], %{{.+}})
// CHECK: %[[ACC:.+]] = stablehlo.add %[[SCAN]], %{{.+}} : tensor<4xf64>
// CHECK: stablehlo.slice %[[ACC]] [3:4] : (tensor<4xf64>) -> tensor<1xf64>

// -----

// Under a lane, with the lane's own axis starting past zero too.
func.func private @rows(%x: memref<8x8xf64>, %y: memref<8xf64>) {
  %c0 = arith.constant 0.0 : f64
  affine.parallel (%i) = (2) to (8) {
    %1 = affine.for %k = 1 to 8 iter_args(%acc = %c0) -> (f64) {
      %3 = affine.load %x[%i, %k] : memref<8x8xf64>
      %4 = arith.addf %acc, %3 : f64
      affine.yield %4 : f64
    }
    affine.store %1, %y[%i] : memref<8xf64>
  }
  return
}

// CHECK-LABEL: func.func private @rows_raised(
// CHECK: stablehlo.slice %{{.+}} [0:6, 6:7] : (tensor<6x7xf64>) -> tensor<6x1xf64>
