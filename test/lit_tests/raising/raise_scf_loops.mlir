// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A grid-stride reduction: an scf.for with uniform runtime bounds raises as
// a stablehlo.while whose counter is a rank-0 scalar of the loop's integer
// type, carrying the per-lane accumulator as a batched tensor.
func.func @scfred(%out: memref<100xf64, 1>, %in: memref<?xf64, 1>, %nb: memref<i32, 1>) {
  %z = arith.constant 0.0 : f64
  %c100 = arith.constant 100 : i32
  affine.parallel (%t) = (0) to (100) {
    %n = affine.load %nb[] : memref<i32, 1>
    %sum = scf.for %j = %c100 to %n step %c100 iter_args(%acc = %z) -> (f64) : i32 {
      %ti = arith.index_castui %t : index to i32
      %idx32 = arith.addi %j, %ti : i32
      %idx = arith.index_cast %idx32 : i32 to index
      %v = memref.load %in[%idx] : memref<?xf64, 1>
      %a = arith.addf %acc, %v : f64
      scf.yield %a : f64
    }
    affine.store %sum, %out[%t] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @scfred_raised(
// CHECK: %[[W:[0-9]+]]:5 = stablehlo.while(%[[IV:[a-zA-Z0-9_]+]] = %{{[^,]+}}, %[[ACC:[a-zA-Z0-9_]+]] = %{{[^,]+}}, %{{.+}}) : tensor<i32>, tensor<100xf64>, tensor<100xf64>, tensor<?xf64>, tensor<i32>
// CHECK: cond {
// CHECK: stablehlo.compare LT, %[[IV]], %{{.+}} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK: } do {
// CHECK: "stablehlo.gather"
// CHECK: arith.addf %[[ACC]], %{{.+}} : tensor<100xf64>
// CHECK: stablehlo.add %[[IV]], %{{.+}} : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %[[W]]#2, %{{.+}} : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>

// -----

// Ping-pong buffers: branches yielding one of two buffers expand into a
// branch per access, so loads select between the carried tensors and stores
// mask into each, with no frozen alias.
#even = affine_set<(d0) : (d0 mod 2 == 0)>
func.func @pingpong(%a: memref<100xf64, 1>, %b: memref<100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %s = 0 to %ni {
    affine.parallel (%i) = (0) to (100) {
      %src = affine.if #even(%s) -> memref<100xf64, 1> {
        affine.yield %a : memref<100xf64, 1>
      } else {
        affine.yield %b : memref<100xf64, 1>
      }
      %dst = affine.if #even(%s) -> memref<100xf64, 1> {
        affine.yield %b : memref<100xf64, 1>
      } else {
        affine.yield %a : memref<100xf64, 1>
      }
      %v = affine.load %src[%i] : memref<100xf64, 1>
      %w = arith.mulf %v, %v : f64
      affine.store %w, %dst[%i] : memref<100xf64, 1>
    }
  } {enzymexla.parallel}
  return
}

// CHECK-LABEL: func.func private @pingpong_raised(
// CHECK: stablehlo.while(%{{[^,]+}} = %{{[^,]+}}, %{{[^,]+}} = %arg0, %{{[^,]+}} = %arg1, %{{.+}}) : tensor<i64>, tensor<100xf64>, tensor<100xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK: } do {
// CHECK: stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<100xi1>, tensor<100xf64>
// CHECK: arith.mulf
// CHECK-COUNT-2: stablehlo.dynamic_update_slice

// -----

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
