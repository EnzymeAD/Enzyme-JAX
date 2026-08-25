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

