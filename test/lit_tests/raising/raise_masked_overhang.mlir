// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A 32-lane batch storing s[t] under a dynamic guard t < n into a 25-element
// buffer: lanes past the end could only have stored out of bounds, so their
// mask is necessarily false and both the update and the mask slice down to
// the buffer extent.
func.func @overhang(%src: memref<32xf64, 1>, %out: memref<25xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%t) = (0) to (32) {
    affine.if affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>(%t)[%ni] {
      %v = affine.load %src[%t] : memref<32xf64, 1>
      affine.store %v, %out[%t] : memref<25xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @overhang_raised(
// CHECK-SAME: %[[SRC:.+]]: tensor<32xf64>, %[[OUT:.+]]: tensor<25xf64>, %[[N:.+]]: tensor<i64>
// CHECK: %[[MASK:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
// CHECK: %[[UPD:.+]] = stablehlo.slice %{{.+}} [0:25] : (tensor<32xf64>) -> tensor<25xf64>
// CHECK-NEXT: %[[MSK:.+]] = stablehlo.slice %[[MASK]] [0:25] : (tensor<32xi1>) -> tensor<25xi1>
// CHECK-NEXT: %[[PREV:.+]] = stablehlo.dynamic_slice %[[OUT]], %{{.+}}, sizes = [25] : (tensor<25xf64>, tensor<i64>) -> tensor<25xf64>
// CHECK: stablehlo.select %[[MSK]], %{{.+}}, %{{.+}} : tensor<25xi1>, tensor<25xf64>
// CHECK: stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}} : (tensor<32xf64>, tensor<25xf64>, tensor<i64>) -> tensor<32xf64>

// -----

// A masked single-element store into a buffer of dynamic extent: the update
// always fits a dynamic dimension, so the masked path raises instead of
// reporting an unfit update.
func.func @dynbuf(%out: memref<?xf64, 1>, %in: memref<16xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%t) = (0) to (32) {
    affine.if affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>(%t)[%ni] {
      %v = affine.load %in[5] : memref<16xf64, 1>
      affine.store %v, %out[7] : memref<?xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @dynbuf_raised(
// CHECK-SAME: %[[DOUT:.+]]: tensor<?xf64>, %[[IN:.+]]: tensor<16xf64>, %[[DN:.+]]: tensor<i64>
// CHECK: stablehlo.reduce({{.+}} init: {{.+}}) applies stablehlo.or across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
// CHECK: stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}} : (tensor<?xf64>, tensor<1xf64>, tensor<i64>) -> tensor<?xf64>
