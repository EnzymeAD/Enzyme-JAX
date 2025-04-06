// RUN: enzymexlamlir-opt %s '--raise-affine-to-stablehlo=enable_lockstep_for=false prefer_while_raising=false' | FileCheck %s

// CHECK-LABEL:  func.func private @foo_raised(
// CHECK-SAME:       %[[ARG:.*]]: tensor<9x32x16xf64>
// CHECK:      return
// CHECK-NOT:  %[[ARG]]
// CHECK-SAME: tensor<9x32x16xf64>

module {
  func.func private @foo(%arg0: memref<9x32x16xf64, 1>) {
    %cst = arith.constant 5.000000e-01 : f64
    affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0) to (32, 2, 2, 2, 16) {
      affine.for %arg7 = 0 to 2 {
        affine.store %cst, %arg0[-%arg7 + 6, %arg2, %arg6] : memref<9x32x16xf64, 1>
      }
    }
    return
  }
}
