// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

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
