// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=prefer_while_raising=false | FileCheck %s

// CHECK-LABEL: func.func private @waw_unroll_raised
func.func @waw_unroll(%arg0: memref<1xf64>) {
  %cst = arith.constant 2.0 : f64
  // CHECK-NOT: stablehlo.while
  // CHECK: stablehlo.dynamic_update_slice
  // CHECK: stablehlo.dynamic_update_slice
  affine.for %arg2 = 0 to 2 {
    affine.store %cst, %arg0[0] : memref<1xf64>
  }
  return
}
