// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-stablehlo --canonicalize --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

// A reduction whose yielded value does not depend on the loop IV still folds
// one copy of the value per iteration: summing 1.0 over 3 iterations is 3.0
// (https://github.com/EnzymeAD/Reactant.jl/issues/3492).

// CHECK-LABEL:   func.func private @reduce_invariant_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1xf64>) -> tensor<1xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<1xf64>
// CHECK:           return %[[VAL_1]] : tensor<1xf64>
// CHECK:         }
func.func private @reduce_invariant(%arg0: memref<1xf64, 1>) {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = affine.parallel (%arg1) = (0) to (3) reduce ("addf") -> (f64) {
    affine.yield %cst : f64
  }
  affine.store %0, %arg0[0] : memref<1xf64, 1>
  return
}

// -----

// Invariant along one of two IVs of the same parallel op: every (i, j) pair
// contributes, so the k-lane is replicated along the j axis before reducing.

// CHECK-LABEL:   func.func private @reduce_partially_invariant_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1xf64>, %[[VAL_1:.*]]: tensor<5xf64>) -> (tensor<1xf64>, tensor<5xf64>) {
// CHECK-DAG:       %[[VAL_2:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_3]]) applies stablehlo.add across dimensions = [0] : (tensor<5xf64>, tensor<f64>) -> tensor<f64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_2]] : tensor<f64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<f64>) -> tensor<1xf64>
// CHECK:           return %[[VAL_6]], %[[VAL_1]] : tensor<1xf64>, tensor<5xf64>
// CHECK:         }
func.func private @reduce_partially_invariant(%arg0: memref<1xf64, 1>, %arg1: memref<5xf64, 1>) {
  %0 = affine.parallel (%k, %j) = (0, 0) to (5, 4) reduce ("addf") -> (f64) {
    %1 = affine.load %arg1[%k] : memref<5xf64, 1>
    affine.yield %1 : f64
  }
  affine.store %0, %arg0[0] : memref<1xf64, 1>
  return
}

// -----

// An invariant inner reduction inside an outer parallel loop: the yielded
// value varies along the outer IV only, and must still be summed over the
// inner trip count.

// CHECK-LABEL:   func.func private @reduce_invariant_nested_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10xf64>, %[[VAL_1:.*]]: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<10xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.multiply %[[VAL_1]], %[[VAL_2]] : tensor<10xf64>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10xf64>, tensor<10xf64>
// CHECK:         }
func.func private @reduce_invariant_nested(%arg0: memref<10xf64, 1>, %arg1: memref<10xf64, 1>) {
  affine.parallel (%i) = (0) to (10) {
    %0 = affine.parallel (%k) = (0) to (3) reduce ("addf") -> (f64) {
      %1 = affine.load %arg1[%i] : memref<10xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i] : memref<10xf64, 1>
  }
  return
}

// -----

// Multiplication: an invariant factor is raised to the trip count.

// CHECK-LABEL:   func.func private @reduce_invariant_mulf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1xf64>, %[[VAL_1:.*]]: tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<4.000000e+00> : tensor<1xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.power %[[VAL_1]], %[[VAL_2]] : tensor<1xf64>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<1xf64>, tensor<1xf64>
// CHECK:         }
func.func private @reduce_invariant_mulf(%arg0: memref<1xf64, 1>, %arg1: memref<1xf64, 1>) {
  %v = affine.load %arg1[0] : memref<1xf64, 1>
  %0 = affine.parallel (%arg2) = (0) to (4) reduce ("mulf") -> (f64) {
    affine.yield %v : f64
  }
  affine.store %0, %arg0[0] : memref<1xf64, 1>
  return
}
