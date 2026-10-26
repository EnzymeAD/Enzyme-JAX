// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// A loop that runs once is its body at the lower bound, even one the
// parallelization could not take (it calls out).

llvm.func @use(f64)
func.func @once(%m: memref<?xf64>, %x: f64) -> f64 {
  %r = affine.for %i = 0 to 1 iter_args(%acc = %x) -> f64 {
    %v = affine.load %m[%i] : memref<?xf64>
    llvm.call @use(%v) : (f64) -> ()
    %s = arith.addf %acc, %v : f64
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK-LABEL:   func.func @once(
// CHECK-SAME:                    %[[ARG0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                    %[[ARG1:[a-z0-9]+]]: f64) -> f64 {
// CHECK:           %[[LOAD_0:.*]] = affine.load %[[ARG0]][0] : memref<?xf64>
// CHECK:           llvm.call @use(%[[LOAD_0]]) : (f64) -> ()
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[ARG1]], %[[LOAD_0]] : f64
// CHECK:           return %[[ADDF_0]] : f64
// CHECK:         }

// -----

// A loop that never runs gives its initial values.

llvm.func @use(f64)
func.func @never(%m: memref<?xf64>, %x: f64) -> f64 {
  %r = affine.for %i = 4 to 2 iter_args(%acc = %x) -> f64 {
    %v = affine.load %m[%i] : memref<?xf64>
    llvm.call @use(%v) : (f64) -> ()
    %s = arith.addf %acc, %v : f64
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK-LABEL:   func.func @never(
// CHECK-SAME:                     %[[ARG0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                     %[[ARG1:[a-z0-9]+]]: f64) -> f64 {
// CHECK:           return %[[ARG1]] : f64
// CHECK:         }

// -----

// A loop whose upper bound is at most the lower plus one runs once or not at
// all: its body under the condition that it runs, its initial values
// otherwise.

llvm.func @use(f64)
func.func @at_most_once(%m: memref<?xf64>, %x: f64, %n: index, %k: index) -> f64 {
  %r = affine.for %i = %k to min affine_map<(d0)[s0] -> (d0 + 1, s0)>(%k)[%n] iter_args(%acc = %x) -> f64 {
    %v = affine.load %m[%i] : memref<?xf64>
    llvm.call @use(%v) : (f64) -> ()
    %s = arith.addf %acc, %v : f64
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK: #[[$ATTR_0:.+]] = affine_set<()[s0, s1] : (s1 - s0 - 1 >= 0)>
// CHECK-LABEL:   func.func @at_most_once(
// CHECK-SAME:                            %[[ARG0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                            %[[ARG1:[a-z0-9]+]]: f64,
// CHECK-SAME:                            %[[ARG2:[a-z0-9]+]]: index,
// CHECK-SAME:                            %[[ARG3:[a-z0-9]+]]: index) -> f64 {
// CHECK:           %[[IF_0:.*]] = affine.if #[[$ATTR_0]](){{\[}}%[[ARG3]], %[[ARG2]]] -> f64 {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG0]][symbol(%[[ARG3]])] : memref<?xf64>
// CHECK:             llvm.call @use(%[[LOAD_0]]) : (f64) -> ()
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[ARG1]], %[[LOAD_0]] : f64
// CHECK:             affine.yield %[[ADDF_0]] : f64
// CHECK:           } else {
// CHECK:             affine.yield %[[ARG1]] : f64
// CHECK:           }
// CHECK:           return %[[IF_0]] : f64
// CHECK:         }

// -----

// One that may run twice stays a loop.

llvm.func @use(f64)
func.func @twice(%m: memref<?xf64>, %x: f64, %n: index, %k: index) -> f64 {
  %r = affine.for %i = %k to min affine_map<(d0)[s0] -> (d0 + 2, s0)>(%k)[%n] iter_args(%acc = %x) -> f64 {
    %v = affine.load %m[%i] : memref<?xf64>
    llvm.call @use(%v) : (f64) -> ()
    %s = arith.addf %acc, %v : f64
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0, s1] -> (s1 + 2, s0)>
// CHECK-LABEL:   func.func @twice(
// CHECK-SAME:                     %[[ARG0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                     %[[ARG1:[a-z0-9]+]]: f64,
// CHECK-SAME:                     %[[ARG2:[a-z0-9]+]]: index,
// CHECK-SAME:                     %[[ARG3:[a-z0-9]+]]: index) -> f64 {
// CHECK:           %[[FOR_0:.*]] = affine.for %[[VAL_0:.*]] = %[[ARG3]] to min #[[$ATTR_1]](){{\[}}%[[ARG2]], %[[ARG3]]] iter_args(%[[VAL_1:.*]] = %[[ARG1]]) -> (f64) {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG0]]{{\[}}%[[VAL_0]]] : memref<?xf64>
// CHECK:             llvm.call @use(%[[LOAD_0]]) : (f64) -> ()
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[LOAD_0]] : f64
// CHECK:             affine.yield %[[ADDF_0]] : f64
// CHECK:           }
// CHECK:           return %[[FOR_0]] : f64
// CHECK:         }
