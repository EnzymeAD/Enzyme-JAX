// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file --verify-diagnostics | FileCheck %s

// Every lane along %t stores its own value to out[%e]: the program leaves
// the winner undefined, so lane 0's value is a sound refinement.
func.func @racy(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (8, 32) {
    %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
    // expected-warning @below {{racy store: the stored value varies along a parallel axis the destination does not index; raising lane 0's write}}
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
    // expected-warning @below {{racy store: the stored value varies along a parallel axis the destination does not index; raising one lane's write}}
    affine.store %v, %out[%e * 2] : memref<16xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @racy_scatter_raised(
// CHECK: %[[G:.+]] = "stablehlo.gather"
// CHECK: %[[S:.+]] = stablehlo.slice %[[G]] [0:8, 0:1] : (tensor<8x32xf64>) -> tensor<8x1xf64>
// CHECK: %[[R:.+]] = stablehlo.reshape %[[S]] : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK: "stablehlo.scatter"
