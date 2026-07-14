// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-stablehlo --canonicalize --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

// CHECK-LABEL:   func.func private @reduce_addf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xf64>, %[[VAL_1:.*]]: tensor<5x10x20xf64>) -> (tensor<10x20xf64>, tensor<5x10x20xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies stablehlo.add across dimensions = [0] : (tensor<5x10x20xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xf64>, tensor<5x10x20xf64>
// CHECK:         }
func.func private @reduce_addf(%arg0: memref<10x20xf64, 1>, %arg1: memref<5x10x20xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("addf") -> (f64) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xf64, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_mulf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xf64>, %[[VAL_1:.*]]: tensor<5x10x20xf64>) -> (tensor<10x20xf64>, tensor<5x10x20xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies stablehlo.multiply across dimensions = [0] : (tensor<5x10x20xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xf64>, tensor<5x10x20xf64>
// CHECK:         }
func.func private @reduce_mulf(%arg0: memref<10x20xf64, 1>, %arg1: memref<5x10x20xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("mulf") -> (f64) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xf64, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_andi_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xi1>, %[[VAL_1:.*]]: tensor<5x10x20xi1>) -> (tensor<10x20xi1>, tensor<5x10x20xi1>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies stablehlo.and across dimensions = [0] : (tensor<5x10x20xi1>, tensor<i1>) -> tensor<10x20xi1>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xi1>, tensor<5x10x20xi1>
// CHECK:         }
func.func private @reduce_andi(%arg0: memref<10x20xi1, 1>, %arg1: memref<5x10x20xi1, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("andi") -> (i1) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xi1, 1>
      affine.yield %1 : i1
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xi1, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_xori_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xi1>, %[[VAL_1:.*]]: tensor<5x10x20xi1>) -> (tensor<10x20xi1>, tensor<5x10x20xi1>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies stablehlo.xor across dimensions = [0] : (tensor<5x10x20xi1>, tensor<i1>) -> tensor<10x20xi1>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xi1>, tensor<5x10x20xi1>
// CHECK:         }
func.func private @reduce_xori(%arg0: memref<10x20xi1, 1>, %arg1: memref<5x10x20xi1, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("xori") -> (i1) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xi1, 1>
      affine.yield %1 : i1
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xi1, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_maximumf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xf64>, %[[VAL_1:.*]]: tensor<5x10x20xf64>) -> (tensor<10x20xf64>, tensor<5x10x20xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies stablehlo.maximum across dimensions = [0] : (tensor<5x10x20xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xf64>, tensor<5x10x20xf64>
// CHECK:         }
func.func private @reduce_maximumf(%arg0: memref<10x20xf64, 1>, %arg1: memref<5x10x20xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("maximumf") -> (f64) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xf64, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_maxnumf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xf64>, %[[VAL_1:.*]]: tensor<5x10x20xf64>) -> (tensor<10x20xf64>, tensor<5x10x20xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) across dimensions = [0] : (tensor<5x10x20xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK:            reducer(%[[VAL_4:.*]]: tensor<f64>, %[[VAL_5:.*]]: tensor<f64>)  {
// CHECK:             %[[VAL_6:.*]] = stablehlo.is_finite %[[VAL_4]] : (tensor<f64>) -> tensor<i1>
// CHECK:             %[[VAL_7:.*]] = stablehlo.not %[[VAL_6]] : tensor<i1>
// CHECK:             %[[VAL_8:.*]] = chlo.is_inf %[[VAL_4]] : tensor<f64> -> tensor<i1>
// CHECK:             %[[VAL_9:.*]] = stablehlo.not %[[VAL_8]] : tensor<i1>
// CHECK:             %[[VAL_10:.*]] = stablehlo.and %[[VAL_7]], %[[VAL_9]] : tensor<i1>
// CHECK:             %[[VAL_11:.*]] = stablehlo.maximum %[[VAL_4]], %[[VAL_5]] : tensor<f64>
// CHECK:             %[[VAL_12:.*]] = stablehlo.select %[[VAL_10]], %[[VAL_5]], %[[VAL_11]] : tensor<i1>, tensor<f64>
// CHECK:             stablehlo.return %[[VAL_12]] : tensor<f64>
// CHECK:           }
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xf64>, tensor<5x10x20xf64>
// CHECK:         }
func.func private @reduce_maxnumf(%arg0: memref<10x20xf64, 1>, %arg1: memref<5x10x20xf64, 1>) {
  %cst = arith.constant 0.0 : f64
  %dummy = math.isnan %cst : f64
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("maxnumf") -> (f64) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xf64, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_minimumf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xf64>, %[[VAL_1:.*]]: tensor<5x10x20xf64>) -> (tensor<10x20xf64>, tensor<5x10x20xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies stablehlo.minimum across dimensions = [0] : (tensor<5x10x20xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xf64>, tensor<5x10x20xf64>
// CHECK:         }
func.func private @reduce_minimumf(%arg0: memref<10x20xf64, 1>, %arg1: memref<5x10x20xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("minimumf") -> (f64) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xf64, 1>
  }
  return
}

// -----

// CHECK-LABEL:   func.func private @reduce_minnumf_raised(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<10x20xf64>, %[[VAL_1:.*]]: tensor<5x10x20xf64>) -> (tensor<10x20xf64>, tensor<5x10x20xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) across dimensions = [0] : (tensor<5x10x20xf64>, tensor<f64>) -> tensor<10x20xf64>
// CHECK:            reducer(%[[VAL_4:.*]]: tensor<f64>, %[[VAL_5:.*]]: tensor<f64>)  {
// CHECK:             %[[VAL_6:.*]] = stablehlo.is_finite %[[VAL_4]] : (tensor<f64>) -> tensor<i1>
// CHECK:             %[[VAL_7:.*]] = stablehlo.not %[[VAL_6]] : tensor<i1>
// CHECK:             %[[VAL_8:.*]] = chlo.is_inf %[[VAL_4]] : tensor<f64> -> tensor<i1>
// CHECK:             %[[VAL_9:.*]] = stablehlo.not %[[VAL_8]] : tensor<i1>
// CHECK:             %[[VAL_10:.*]] = stablehlo.and %[[VAL_7]], %[[VAL_9]] : tensor<i1>
// CHECK:             %[[VAL_11:.*]] = stablehlo.minimum %[[VAL_4]], %[[VAL_5]] : tensor<f64>
// CHECK:             %[[VAL_12:.*]] = stablehlo.select %[[VAL_10]], %[[VAL_5]], %[[VAL_11]] : tensor<i1>, tensor<f64>
// CHECK:             stablehlo.return %[[VAL_12]] : tensor<f64>
// CHECK:           }
// CHECK:           return %[[VAL_3]], %[[VAL_1]] : tensor<10x20xf64>, tensor<5x10x20xf64>
// CHECK:         }
func.func private @reduce_minnumf(%arg0: memref<10x20xf64, 1>, %arg1: memref<5x10x20xf64, 1>) {
  %cst = arith.constant 0.0 : f64
  %dummy = math.isnan %cst : f64
  affine.parallel (%i, %j) = (0, 0) to (10, 20) {
    %0 = affine.parallel (%k) = (0) to (5) reduce ("minnumf") -> (f64) {
      %1 = affine.load %arg1[%k, %i, %j] : memref<5x10x20xf64, 1>
      affine.yield %1 : f64
    }
    affine.store %0, %arg0[%i, %j] : memref<10x20xf64, 1>
  }
  return
}
