// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// A loop that runs once carries nothing from one iteration to the next, so
// the accesses in it are never compared, however they conflict: the loop is
// promoted. One that runs twice keeps the store it reads back serial.

func.func @once(%m: memref<?xf64>, %x: f64, %n: index) {
  affine.for %i = 0 to 1 {
    affine.store %x, %m[0] : memref<?xf64>
    %v = affine.load %m[0] : memref<?xf64>
    %w = arith.addf %v, %x : f64
    affine.store %w, %m[symbol(%n)] : memref<?xf64>
  }
  return
}

// CHECK-LABEL:   func.func @once(
// CHECK-SAME:                    %[[ARG0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                    %[[ARG1:[a-z0-9]+]]: f64,
// CHECK-SAME:                    %[[ARG2:[a-z0-9]+]]: index) {
// CHECK:           affine.store %[[ARG1]], %[[ARG0]][0] : memref<?xf64>
// CHECK:           %[[LOAD_0:.*]] = affine.load %[[ARG0]][0] : memref<?xf64>
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG1]] : f64
// CHECK:           affine.store %[[ADDF_0]], %[[ARG0]][symbol(%[[ARG2]])] : memref<?xf64>
// CHECK:           return
// CHECK:         }

func.func @twice(%m: memref<?xf64>, %x: f64, %n: index) {
  affine.for %i = 0 to 2 {
    affine.store %x, %m[0] : memref<?xf64>
    %v = affine.load %m[0] : memref<?xf64>
    %w = arith.addf %v, %x : f64
    affine.store %w, %m[symbol(%n)] : memref<?xf64>
  }
  return
}

// CHECK-LABEL:   func.func @twice(
// CHECK-SAME:                     %[[ARG0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                     %[[ARG1:[a-z0-9]+]]: f64,
// CHECK-SAME:                     %[[ARG2:[a-z0-9]+]]: index) {
// CHECK:           affine.for %[[VAL_0:.*]] = 0 to 2 {
// CHECK:             affine.store %[[ARG1]], %[[ARG0]][0] : memref<?xf64>
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG0]][0] : memref<?xf64>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG1]] : f64
// CHECK:             affine.store %[[ADDF_0]], %[[ARG0]][symbol(%[[ARG2]])] : memref<?xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }
