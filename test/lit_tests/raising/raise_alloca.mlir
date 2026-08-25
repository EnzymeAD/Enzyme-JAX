// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// Scratch memory raises as a zero splat tensor materialized where the
// alloca sits; writes then update it like any other buffer.
func.func @scratch(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  affine.parallel (%i) = (0) to (10) {
    %c = arith.constant 3.0 : f64
    affine.store %c, %tmp[%i] : memref<10xf64>
  }
  affine.parallel (%i) = (0) to (10) {
    %v = affine.load %tmp[9 - %i] : memref<10xf64>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @scratch_raised(
// CHECK-DAG: stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK-DAG: stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: stablehlo.reverse

// -----

// Reads that happen before any write observe zeros.
func.func @zeroinit(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  affine.parallel (%i) = (0) to (10) {
    %v = affine.load %tmp[%i] : memref<10xf64>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @zeroinit_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<10xf64>
// CHECK: stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: stablehlo.dynamic_update_slice %[[OUT]], %{{.+}}, %{{.+}} : (tensor<10xf64>, tensor<10xf64>, tensor<i64>) -> tensor<10xf64>

// -----

// Scratch updated inside a loop is carried through the raised while, so
// writes from one iteration are visible in the next.
func.func @accum(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  affine.for %t = 0 to 4 {
    affine.parallel (%i) = (0) to (10) {
      %v = affine.load %tmp[%i] : memref<10xf64>
      %c = arith.constant 1.0 : f64
      %s = arith.addf %v, %c : f64
      affine.store %s, %tmp[%i] : memref<10xf64>
    }
  }
  affine.parallel (%i) = (0) to (10) {
    %v = affine.load %tmp[%i] : memref<10xf64>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @accum_raised(
// CHECK: %[[ZSPLAT:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: %[[W:.+]]:3 = stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %[[SCRATCH:.+]] = %[[ZSPLAT]]) : tensor<i64>, tensor<10xf64>, tensor<10xf64>
// CHECK: } do {
// CHECK: stablehlo.dynamic_update_slice %[[SCRATCH]], %{{.+}}, %{{.+}} : (tensor<10xf64>, tensor<10xf64>, tensor<i64>) -> tensor<10xf64>
