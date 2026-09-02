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

// The scratch reaches one access directly as a plain memref.load next to the
// viewed accesses: it moves to the flat buffer with its indexing linearized
// instead of keeping the whole chain opaque.
func.func @direct_mix(%out: memref<32xf64, 1>) {
  affine.parallel (%i) = (0) to (32) {
    %scr = memref.alloca() : memref<32xf64>
    %p = "enzymexla.memref2pointer"(%scr) : (memref<32xf64>) -> !llvm.ptr<3>
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
    %c = arith.constant 3.0 : f64
    affine.store %c, %v[%i] : memref<?xf64, 3>
    %idx = arith.constant 0 : index
    %r = memref.load %scr[%idx] : memref<32xf64>
    affine.store %r, %out[%i] : memref<32xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @direct_mix_raised(

// -----

// Views can reach the scratch through an address-space cast and a
// constant-offset gep; the accumulated byte offset folds into the access
// indices of the flat buffer.
func.func @offsetview(%out: memref<8xf64, 1>, %in: memref<8xf64, 1>) {
  %scr = memref.alloca() : memref<16xf64>
  %ptr = "enzymexla.memref2pointer"(%scr) : (memref<16xf64>) -> !llvm.ptr<3>
  %gp = llvm.addrspacecast %ptr : !llvm.ptr<3> to !llvm.ptr
  %off = llvm.getelementptr inbounds %gp[64] : (!llvm.ptr) -> !llvm.ptr, i8
  affine.parallel (%t) = (0) to (8) {
    %view = "enzymexla.pointer2memref"(%off) : (!llvm.ptr) -> memref<?xf64>
    %v = affine.load %in[%t] : memref<8xf64, 1>
    affine.store %v, %view[%t] : memref<?xf64>
    %r = affine.load %view[7 - %t] : memref<?xf64>
    affine.store %r, %out[%t] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @offsetview_raised(
// CHECK: stablehlo.reverse
