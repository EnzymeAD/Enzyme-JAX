// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A tid==0 broadcast store: the mask covers the thread axis but the store
// does not index it, and the stored value is invariant along it, so the mask
// or-reduces over that axis.
func.func @lane0(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
  affine.parallel (%t, %j) = (0, 0) to (32, 16) {
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      %v = affine.load %in[%j] : memref<16xf64, 1>
      affine.store %v, %out[%j] : memref<16xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @lane0_raised(
// CHECK: stablehlo.reduce(%{{.+}} init: %{{.+}}) applies stablehlo.or across dimensions =
// CHECK: stablehlo.select

// -----

// The block-wide pattern: scratch filled by every lane, one lane broadcasting
// a scalar, a barrier, then every lane combining both. The barrier is a
// no-op in raised form and the broadcast mask or-reduces over the thread
// axis it does not index.
func.func @bcast(%out: memref<?xf64, 1>, %in: memref<?xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %e = 0 to %ni {
    %scr = memref.alloca() : memref<32xf64>
    %b = memref.alloca() : memref<1xf64>
    affine.parallel (%t) = (0) to (32) {
      %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
      affine.store %v, %scr[%t] : memref<32xf64>
      affine.if affine_set<(d0) : (d0 == 0)>(%t) {
        %s = affine.load %in[%e] : memref<?xf64, 1>
        affine.store %s, %b[0] : memref<1xf64>
      }
      "enzymexla.barrier"(%t) : (index) -> ()
      %x = affine.load %scr[%t] : memref<32xf64>
      %y = affine.load %b[0] : memref<1xf64>
      %m = arith.mulf %x, %y : f64
      affine.store %m, %out[%e * 32 + %t] : memref<?xf64, 1>
    }
  } {enzymexla.parallel}
  return
}

// CHECK-LABEL: func.func private @bcast_raised(
// CHECK: stablehlo.while
// CHECK-SAME: attributes {enzymexla.parallel}
// CHECK-NOT: enzymexla.barrier
// CHECK: %[[RED:.+]] = stablehlo.reduce(%{{.+}} init: %{{.+}}) applies stablehlo.or across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
// CHECK: stablehlo.select %[[RED]], %{{.+}}, %{{.+}} : tensor<i1>, tensor<f64>

// -----

// The non-affine store path: a mask axis the scatter grid does not carry
// (the tid==0 guard) or-reduces instead of producing a rank-mismatched
// broadcast.
func.func @bcast_scatter(%out: memref<100xf64, 1>, %in: memref<100xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (4, 32) {
    %v = affine.load %in[%e * 25] : memref<100xf64, 1>
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      %ei = arith.index_castui %e : index to i64
      %e2 = arith.muli %ei, %ei : i64
      %idx = arith.index_cast %e2 : i64 to index
      memref.store %v, %out[%idx] : memref<100xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @bcast_scatter_raised(
// CHECK: stablehlo.reduce{{.*}}applies stablehlo.or across dimensions
// CHECK: "stablehlo.scatter"
