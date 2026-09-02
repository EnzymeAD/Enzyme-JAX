// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel,raise-affine-to-stablehlo,enzyme-hlo-opt{max_constant_expansion=0})" | FileCheck %s

// A pointer round trip into shared memory of a static scratch buffer: the
// shape cast folds into the accesses under canonicalization, leaving only the
// memory space cast, which the raising strips.
func.func @roundtrip(%in: memref<10xf64, 1>, %out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %p = "enzymexla.memref2pointer"(%tmp) : (memref<10xf64>) -> !llvm.ptr<3>
  %view = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
  affine.parallel (%i) = (0) to (10) {
    %v = affine.load %in[%i] : memref<10xf64, 1>
    affine.store %v, %view[%i] : memref<?xf64, 3>
    %r = affine.load %view[9 - %i] : memref<?xf64, 3>
    affine.store %r, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK:  func.func private @roundtrip_raised(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>) {
// CHECK-NEXT:    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<10xf64>
// CHECK-NEXT:    return %arg0, %0 : tensor<10xf64>, tensor<10xf64>
// CHECK-NEXT:  }
