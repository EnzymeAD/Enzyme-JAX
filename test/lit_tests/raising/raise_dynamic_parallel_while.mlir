// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A parallel axis whose extent is only known at runtime raises as a
// stablehlo.while over the raised body, tagged enzymexla.parallel so
// downstream passes know the iterations are independent.
func.func @dynkern(%out: memref<100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%i) = (0) to (symbol(%ni)) {
    %iv = arith.index_castui %i : index to i64
    %v = arith.sitofp %iv : i64 to f64
    affine.store %v, %out[%i] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @dynkern_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<100xf64>, %[[N:.+]]: tensor<i64>
// CHECK: %[[WHILE:.+]]:3 = stablehlo.while(%[[IV:.+]] = %{{.+}}, %[[BUF:.+]] = %[[OUT]], %{{.+}} = %[[N]]) : tensor<i64>, tensor<100xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK: cond {
// CHECK: stablehlo.compare LT, %[[IV]], %{{.+}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK: } do {
// CHECK: %[[V:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK: stablehlo.dynamic_update_slice %[[BUF]], %[[V]], %[[IV]] : (tensor<100xf64>, tensor<1xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK: return %[[WHILE]]#1, %[[WHILE]]#2 : tensor<100xf64>, tensor<i64>

// -----

// Mixed extents: the static axis stays batched inside the while body, only
// the dynamic axis is peeled into the while.
func.func @mixed(%out: memref<16x100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%e, %j) = (0, 0) to (symbol(%ni), 16) {
    %iv = arith.index_castui %j : index to i64
    %v = arith.sitofp %iv : i64 to f64
    affine.store %v, %out[%j, %e] : memref<16x100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @mixed_raised(
// CHECK: %[[W:.+]]:3 = stablehlo.while(%[[IV:.+]] = %{{.+}}, %[[BUF:.+]] = %{{.+}}, %{{.+}} = %{{.+}}) : tensor<i64>, tensor<16x100xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK: } do {
// CHECK: %[[UPD:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [0] : (tensor<16xf64>) -> tensor<16x1xf64>
// CHECK: stablehlo.dynamic_update_slice %[[BUF]], %[[UPD]], %{{.+}}, %[[IV]] : (tensor<16x100xf64>, tensor<16x1xf64>, tensor<i64>, tensor<i64>) -> tensor<16x100xf64>

// -----

// A dynamic parallel loop outside any raised region is left untouched: the
// peel only applies where raising will consume the result.
func.func @host(%out: memref<100xf64, 1>, %n: index) -> index {
  affine.parallel (%i) = (0) to (%n) {
    %c = arith.constant 1.0 : f64
    affine.store %c, %out[%i] : memref<100xf64, 1>
  }
  return %n : index
}

// CHECK-LABEL: func.func @host(
// CHECK: affine.parallel
// CHECK-NOT: enzymexla.parallel
