// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file --verify-diagnostics | FileCheck %s

// Every lane along %t stores its own value to out[%e]: the program leaves
// the winner undefined, so lane 0's value is a sound refinement.
func.func @racy(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (8, 32) {
    %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
    affine.store %v, %out[%e] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @racy_raised(
// CHECK: %[[G:.+]] = "stablehlo.gather"
// CHECK: %[[S:.+]] = stablehlo.slice %[[G]] [0:8, 0:1] : (tensor<8x32xf64>) -> tensor<8x1xf64>
// CHECK: %[[R:.+]] = stablehlo.reshape %[[S]] : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK: stablehlo.dynamic_update_slice %arg0, %{{.+}}, %{{.+}} : (tensor<8xf64>, tensor<8xf64>, tensor<i64>) -> tensor<8xf64>

// -----

// The same race through the scatter path (a strided destination).
func.func @racy_scatter(%out: memref<16xf64, 1>, %in: memref<?xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (8, 32) {
    %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
    affine.store %v, %out[%e * 2] : memref<16xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @racy_scatter_raised(
// CHECK: %[[G:.+]] = "stablehlo.gather"
// CHECK: %[[S:.+]] = stablehlo.slice %[[G]] [0:8, 0:1] : (tensor<8x32xf64>) -> tensor<8x1xf64>
// CHECK: %[[R:.+]] = stablehlo.reshape %[[S]] : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK: "stablehlo.scatter"

// -----

// Under a guard varying along the racing axis, which lanes write is data
// dependent: no refinement, the store stays unraisable.
func.func @racy_guarded(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>, %qb: memref<i64, 1>) {
  %qi = affine.load %qb[] : memref<i64, 1>
  %q = arith.index_cast %qi : i64 to index
  affine.parallel (%e, %t) = (0, 0) to (8, 32) {
    affine.if affine_set<(d0)[s0] : (d0 - s0 == 0)>(%t)[%q] {
      %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
      // expected-error @below {{affine.store is dependent on less dims than stored value}}
      affine.store %v, %out[%e] : memref<8xf64, 1>
    }
  }
  return
}

