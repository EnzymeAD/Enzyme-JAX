// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// Shared-memory scratch: a static alloca only accessed through a
// shape-and-address-space-changing memref2pointer/pointer2memref round trip
// flattens to one static alloca, which raises as a zero-initialized tensor.
func.func @kern(%out: memref<32xf64, 1>) {
  affine.parallel (%i) = (0) to (32) {
    %a = memref.alloca() : memref<32x1xf64>
    %p = "enzymexla.memref2pointer"(%a) : (memref<32x1xf64>) -> !llvm.ptr<3>
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
    %c = arith.constant 3.0 : f64
    affine.store %c, %v[%i] : memref<?xf64, 3>
    %r = affine.load %v[%i] : memref<?xf64, 3>
    affine.store %r, %out[%i] : memref<32xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @kern_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<32xf64>
// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<32xf64>
// CHECK: stablehlo.dynamic_update_slice %[[ZERO]], %{{.+}} : (tensor<32xf64>, tensor<32xf64>, tensor<i64>) -> tensor<32xf64>

// -----

// A direct affine access moves onto the flat buffer with its map linearized,
// so a store through the alloca stays visible to a read through the view.
func.func @mixed_use(%out: memref<32xf64, 1>) {
  affine.parallel (%i) = (0) to (32) {
    %a = memref.alloca() : memref<32xf64>
    %p = "enzymexla.memref2pointer"(%a) : (memref<32xf64>) -> !llvm.ptr<3>
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
    %c = arith.constant 3.0 : f64
    affine.store %c, %a[%i] : memref<32xf64>
    %r = affine.load %v[%i] : memref<?xf64, 3>
    affine.store %r, %out[%i] : memref<32xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @mixed_use_raised(
// CHECK-SAME: %[[OUT2:.+]]: tensor<32xf64>
// CHECK: %[[Z:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<32xf64>
// CHECK: %[[ST:.+]] = stablehlo.dynamic_update_slice %[[Z]], %{{.+}} : (tensor<32xf64>, tensor<32xf64>, tensor<i64>) -> tensor<32xf64>
// CHECK: %[[RD:.+]] = stablehlo.reshape %[[ST]] : (tensor<32xf64>) -> tensor<32xf64>
// CHECK: %[[BC:.+]] = stablehlo.broadcast_in_dim %[[RD]], dims = [0]
// CHECK: stablehlo.dynamic_update_slice %[[OUT2]], %[[BC]], %{{.+}} : (tensor<32xf64>, tensor<32xf64>, tensor<i64>) -> tensor<32xf64>

// -----

// A block-wide barrier over a batched thread axis is a no-op in raised form
// (whole-tensor updates are already ordered), and a tid==0 broadcast store
// or-reduces its mask over the thread axis it does not index.
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

// A uniform branch choosing between two read-only buffers raises as a select
// of the whole tensors.
func.func @bufsel(%a: memref<100xf64, 1>, %b: memref<100xf64, 1>, %out: memref<100xf64, 1>, %flagbuf: memref<i64, 1>) {
  %f = affine.load %flagbuf[] : memref<i64, 1>
  %fi = arith.index_cast %f : i64 to index
  affine.parallel (%i) = (0) to (100) {
    %buf = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%fi] -> memref<100xf64, 1> {
      affine.yield %a : memref<100xf64, 1>
    } else {
      affine.yield %b : memref<100xf64, 1>
    }
    %v = affine.load %buf[%i] : memref<100xf64, 1>
    affine.store %v, %out[%i] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @bufsel_raised(
// CHECK: stablehlo.select %{{.+}}, %arg0, %arg1 : tensor<i1>, tensor<100xf64>

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
