// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=prefer_while_raising=false | FileCheck %s --check-prefix=UNROLL

// Carried init read through a flattened access; the select rules out lockstep.
func.func private @carried_flattened(%arg0: memref<24x32x32xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %c2_i64 = arith.constant 2 : i64
  affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (2, 16, 2, 16) {
    %x0 = affine.load %arg0[0, %a2 + %a1 * 16, %a4 + %a3 * 16] : memref<24x32x32xf64>
    %r = affine.for %k = 0 to 3 iter_args(%x = %x0) -> (f64) {
      %kk = arith.index_cast %k : index to i64
      %m = arith.cmpi slt, %kk, %c2_i64 : i64
      %y = arith.select %m, %cst, %x : f64
      affine.store %y, %arg0[%k + 1, %a2 + %a1 * 16, %a4 + %a3 * 16] : memref<24x32x32xf64>
      affine.yield %y : f64
    }
  }
  return
}

// CHECK-LABEL: func.func private @carried_flattened_raised(
// CHECK: stablehlo.reshape %{{.+}} : (tensor<32x32xf64>) -> tensor<2x16x2x16xf64>
// CHECK: stablehlo.while
// CHECK-SAME: tensor<2x16x2x16xf64>

// UNROLL-LABEL: func.func private @carried_flattened_raised(
// UNROLL-NOT: affine.for
// UNROLL: return
