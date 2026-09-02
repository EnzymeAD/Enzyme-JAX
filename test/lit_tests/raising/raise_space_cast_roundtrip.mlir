// RUN: enzymexlamlir-opt %s --canonicalize-parallel --raise-affine-to-stablehlo | FileCheck %s

// A pointer round trip into shared memory of a static scratch buffer: the
// shape cast folds into the accesses under canonicalization, leaving only the
// memory space cast, which the raising strips.
func.func @roundtrip(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %p = "enzymexla.memref2pointer"(%tmp) : (memref<10xf64>) -> !llvm.ptr<3>
  %view = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
  affine.parallel (%i) = (0) to (10) {
    %c = arith.constant 2.0 : f64
    affine.store %c, %view[%i] : memref<?xf64, 3>
    %v = affine.load %view[9 - %i] : memref<?xf64, 3>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @roundtrip_raised(
// CHECK-NOT: memref.cast
// CHECK-NOT: memory_space_cast
// CHECK-DAG: stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-DAG: stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: stablehlo.reverse
