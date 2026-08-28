// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A memory_space_cast view of scratch would split one buffer into two SSA
// roots; accesses do not care about the address space, so they retarget to
// the source and the cast drops, letting the scratch raise as usual.
func.func @spacecast(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %view = memref.memory_space_cast %tmp : memref<10xf64> to memref<10xf64, 3>
  affine.parallel (%i) = (0) to (10) {
    %c = arith.constant 2.0 : f64
    affine.store %c, %view[%i] : memref<10xf64, 3>
    %v = affine.load %view[9 - %i] : memref<10xf64, 3>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @spacecast_raised(
// CHECK-NOT: memory_space_cast
// CHECK-DAG: stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-DAG: stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: stablehlo.reverse
