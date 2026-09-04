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

