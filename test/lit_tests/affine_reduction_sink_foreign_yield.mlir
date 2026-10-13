// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

func.func @foreign_yield(%out: memref<1xf64>, %other: memref<1xf64>, %in: memref<8xf64>, %c: index) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf64>
    %s = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%c] -> f64 {
      affine.yield %v : f64
    } else {
      affine.yield %cst : f64
    }
    affine.store %v, %out[0] : memref<1xf64>
    affine.store %s, %other[0] : memref<1xf64>
  }
  return
}

// CHECK-LABEL: func.func @foreign_yield
