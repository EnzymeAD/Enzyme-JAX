// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt --split-input-file | FileCheck %s

// A triangular parallel axis `j = i .. N` has no fixed extent, but over
// i in [0, N) its bounds stay within [0, N): it batches over that interval
// behind a `j >= i` guard instead of being peeled into a masked serial loop.
func.func @triangular(%out: memref<8x8xf64>, %in: memref<8x8xf64>) {
  affine.parallel (%i) = (0) to (8) {
    affine.parallel (%j) = (%i) to (8) {
      %v = affine.load %in[%i, %j] : memref<8x8xf64>
      %w = arith.mulf %v, %v : f64
      affine.store %w, %out[%i, %j] : memref<8x8xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func private @triangular_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<8x8xf64>, %[[IN:.+]]: tensor<8x8xf64>
// CHECK-NOT: stablehlo.while
// CHECK: %[[SQ:.+]] = stablehlo.multiply %[[IN]], %[[IN]] {{.*}}: tensor<8x8xf64>
// CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %[[SQ]], %[[OUT]] {{.*}}: tensor<8x8xi1>, tensor<8x8xf64>
// CHECK: return %[[SEL]], %[[IN]]

// -----

// Lower-triangular: `j = 0 .. i + 1`, a dynamic upper bound in an outer
// induction variable.
func.func @lower(%out: memref<8x8xf64>, %in: memref<8x8xf64>) {
  affine.parallel (%i) = (0) to (8) {
    affine.parallel (%j) = (0) to (%i + 1) {
      %v = affine.load %in[%i, %j] : memref<8x8xf64>
      affine.store %v, %out[%i, %j] : memref<8x8xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func private @lower_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<8x8xf64>, %[[IN:.+]]: tensor<8x8xf64>
// CHECK-NOT: stablehlo.while
// CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %[[IN]], %[[OUT]] {{.*}}: tensor<8x8xi1>, tensor<8x8xf64>
// CHECK: return %[[SEL]], %[[IN]]

// -----

// A band `j = i .. i + 2` is too sparse to batch over its 10-wide hull: it
// stays a masked serial loop over the 2 iterations of a row.
func.func @band(%out: memref<8x10xf64>, %in: memref<8x10xf64>) {
  affine.parallel (%i) = (0) to (8) {
    affine.parallel (%j) = (%i) to (%i + 2) {
      %v = affine.load %in[%i, %j] : memref<8x10xf64>
      affine.store %v, %out[%i, %j] : memref<8x10xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func private @band_raised(
// CHECK: stablehlo.while
