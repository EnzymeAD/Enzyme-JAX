// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A remainder loop whose bounds vary per lane iterates a scalar counter to
// the maximum lane trip count, masking finished lanes: stores mask, and iter
// args keep their value once a lane finishes.
func.func @lanefor(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>) {
  %z = arith.constant 0.0 : f64
  affine.parallel (%t) = (0) to (8) {
    %s = affine.for %j = affine_map<(d0) -> (d0)>(%t) to 8 iter_args(%acc = %z) -> (f64) {
      %v = affine.load %in[%j] : memref<?xf64, 1>
      %a = arith.addf %acc, %v : f64
      affine.yield %a : f64
    }
    affine.store %s, %out[%t] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @lanefor_raised(
// CHECK: stablehlo.reduce{{.*}}applies stablehlo.maximum
// CHECK: stablehlo.while
// CHECK: } do {
// CHECK: %[[ACTIVE:.+]] = stablehlo.compare LT, %{{.+}}, %{{.+}} : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK: stablehlo.select %[[ACTIVE]], %{{.+}}, %{{.+}} : tensor<8xi1>, tensor<8xf64>

// -----

// An scf.for whose lower bound is the lane index takes the same masked path.
func.func @scflane(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>) {
  %z = arith.constant 0.0 : f64
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  affine.parallel (%t) = (0) to (8) {
    %s = scf.for %j = %t to %c8 step %c1 iter_args(%acc = %z) -> (f64) {
      %v = memref.load %in[%j] : memref<?xf64, 1>
      %a = arith.addf %acc, %v : f64
      scf.yield %a : f64
    }
    affine.store %s, %out[%t] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @scflane_raised(
// CHECK: stablehlo.reduce{{.*}}applies stablehlo.maximum
// CHECK: stablehlo.while
// CHECK: } do {
// CHECK: %[[ACTIVE:.+]] = stablehlo.compare LT, %{{.+}}, %{{.+}} : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK: stablehlo.select %[[ACTIVE]], %{{.+}}, %{{.+}} : tensor<8xi1>, tensor<8xf64>
