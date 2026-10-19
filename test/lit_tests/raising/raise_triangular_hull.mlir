// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A triangular nest: the inner axis runs from the outer lane to the end. Raised
// per lane, `j = i + k` skews every access into a diagonal no slice expresses;
// widened to the constant hull [0, 16) behind a `j >= i` guard, both loops are
// plain iota axes and the stores raise to selects over the whole tensor.
func.func @upper(%cov: memref<16x16xf64, 1>, %data: memref<20x16xf64, 1>) {
  %zero = arith.constant 0.0 : f64
  affine.parallel (%i) = (0) to (16) {
    affine.parallel (%j) = (%i) to (16) {
      %s = affine.for %k = 0 to 20 iter_args(%acc = %zero) -> (f64) {
        %a = affine.load %data[%k, %i] : memref<20x16xf64, 1>
        %b = affine.load %data[%k, %j] : memref<20x16xf64, 1>
        %m = arith.mulf %a, %b : f64
        %n = arith.addf %acc, %m : f64
        affine.yield %n : f64
      }
      affine.store %s, %cov[%i, %j] : memref<16x16xf64, 1>
      affine.store %s, %cov[%j, %i] : memref<16x16xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @upper_raised(
// CHECK-NOT: stablehlo.while
// CHECK-NOT: stablehlo.scatter
// CHECK-NOT: stablehlo.gather
// CHECK: stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<16x16xi64>, tensor<16x16xi64>) -> tensor<16x16xi1>
// CHECK: stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<16x16xi1>, tensor<16x16xf64>
// CHECK: stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<16x16xi1>, tensor<16x16xf64>

// -----

// The lower triangle: an upper bound reading the lane, through a multi-result
// (min) map.
func.func @lower(%out: memref<16x16xf64, 1>, %in: memref<16xf64, 1>) {
  affine.parallel (%i) = (0) to (16) {
    affine.parallel (%j) = (0) to (min(16, %i + 1)) {
      %v = affine.load %in[%j] : memref<16xf64, 1>
      affine.store %v, %out[%i, %j] : memref<16x16xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @lower_raised(
// CHECK-NOT: stablehlo.while
// CHECK-NOT: stablehlo.scatter
// CHECK: stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<16x16xi64>, tensor<16x16xi64>) -> tensor<16x16xi1>
// CHECK: stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<16x16xi1>, tensor<16x16xf64>

// -----

// A banded axis: the hull is the whole row, 16 iterations, where the longest
// lane runs 3. Not worth masking 13 of 16 lanes off; it keeps the masked
// per-lane while.
func.func @band(%out: memref<16x16xf64, 1>, %in: memref<16xf64, 1>) {
  affine.parallel (%i) = (0) to (16) {
    affine.parallel (%j) = (%i) to (min(16, %i + 3)) {
      %v = affine.load %in[%j] : memref<16xf64, 1>
      affine.store %v, %out[%i, %j] : memref<16x16xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @band_raised(
// CHECK: stablehlo.reduce{{.*}}applies stablehlo.maximum
// CHECK: stablehlo.while

// -----

// An affine.for whose lower bound is the lane takes the same hull when the
// enclosing loop is a parallel axis, carrying its iter arg through the guard:
// the loop runs the constant hull with a uniform induction variable (the
// load is one dynamic slice, not a per-lane gather) and the guard masks the
// accumulation per lane.
func.func @forlane(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
  %z = arith.constant 0.0 : f64
  affine.parallel (%i) = (0) to (16) {
    %s = affine.for %j = affine_map<(d0) -> (d0)>(%i) to 16 iter_args(%acc = %z) -> (f64) {
      %v = affine.load %in[%j] : memref<16xf64, 1>
      %a = arith.addf %acc, %v : f64
      affine.yield %a : f64
    }
    affine.store %s, %out[%i] : memref<16xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @forlane_raised(
// CHECK-NOT: stablehlo.reduce{{.*}}applies stablehlo.maximum
// CHECK: %[[C16:.+]] = stablehlo.constant dense<16> : tensor<i64>
// CHECK: stablehlo.while
// CHECK: stablehlo.compare LT, %{{.+}}, %[[C16]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK: } do {
// CHECK-NOT: stablehlo.gather
// CHECK: %[[ACTIVE:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<16xi64>, tensor<16xi64>) -> tensor<16xi1>
// CHECK: stablehlo.dynamic_slice %{{.+}}, %{{.+}}, sizes = [1] : (tensor<16xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK: stablehlo.select %[[ACTIVE]], %{{.+}}, %{{.+}} : tensor<16xi1>, tensor<16xf64>
