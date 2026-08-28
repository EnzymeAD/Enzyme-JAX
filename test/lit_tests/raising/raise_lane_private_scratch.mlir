// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false err_if_not_fully_raised=true" | FileCheck %s

// Scratch allocated inside the lane-batched parallel is private to each
// lane: every lane writes its own values at the same indices, so the
// buffer gains one leading dimension per lane axis. Without the
// privatization all lanes would read lane 0's values.

// CHECK-LABEL: @lane_raised
// CHECK-NOT: error

module {
  func.func private @lane(%out: memref<64xf64, 1>, %in: memref<64xf64, 1>) {
    %two = arith.constant 2.0 : f64
    affine.parallel (%e) = (0) to (4) {
      affine.parallel (%t) = (0) to (16) {
        %loc = memref.alloca() : memref<2xf64>
        %v = affine.load %in[%t] : memref<64xf64, 1>
        affine.store %v, %loc[0] : memref<2xf64>
        %v2 = arith.mulf %v, %two : f64
        affine.store %v2, %loc[1] : memref<2xf64>
        %a = affine.load %loc[0] : memref<2xf64>
        %b = affine.load %loc[1] : memref<2xf64>
        %s = arith.addf %a, %b : f64
        affine.store %s, %out[%e * 16 + %t] : memref<64xf64, 1>
      }
    }
    return
  }
}
