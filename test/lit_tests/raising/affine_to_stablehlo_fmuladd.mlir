// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --lower-enzymexla-math --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

// enzymexla.math.fmuladd rides the same route as math.fma: tensorized by the
// affine raising, then split into multiply+add by arith-raise -- stablehlo has
// no fused form, which is the required lowering for strict fma and the
// permitted one for fmuladd alike.

module {
  func.func @fmuladd(%out: memref<8xf64>, %a: memref<8xf64>, %b: memref<8xf64>, %c: memref<8xf64>) {
    affine.parallel (%i) = (0) to (8) {
      %va = affine.load %a[%i] : memref<8xf64>
      %vb = affine.load %b[%i] : memref<8xf64>
      %vc = affine.load %c[%i] : memref<8xf64>
      %r = enzymexla.math.fmuladd %va, %vb, %vc : f64
      affine.store %r, %out[%i] : memref<8xf64>
    }
    return
  }
}

// CHECK:  func.func private @fmuladd_raised(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>, %arg2: tensor<8xf64>, %arg3: tensor<8xf64>) -> (tensor<8xf64>, tensor<8xf64>, tensor<8xf64>, tensor<8xf64>) {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<8xf64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg3 : tensor<8xf64>
// CHECK-NEXT:    return %1, %arg1, %arg2, %arg3 : tensor<8xf64>, tensor<8xf64>, tensor<8xf64>, tensor<8xf64>
// CHECK-NEXT:  }
