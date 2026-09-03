// RUN: enzymexlamlir-opt %s '--raise-affine-to-stablehlo=enable_lockstep_for=false prefer_while_raising=false' | FileCheck %s
// RUN: enzymexlamlir-opt %s '--raise-affine-to-stablehlo=enable_lockstep_for=false prefer_while_raising=false unroll_budget=8' | FileCheck %s --check-prefix=TIGHT
// RUN: enzymexlamlir-opt %s '--raise-affine-to-stablehlo=enable_lockstep_for=false prefer_while_raising=false unroll_budget=-1' | FileCheck %s --check-prefix=UNBOUNDED

// A constant-bound loop whose (nested) unrolled size exceeds the budget
// iterates as a while even when unrolling is preferred.

// CHECK-LABEL: @small_raised
// CHECK-NOT: stablehlo.while

// CHECK-LABEL: @big_raised
// CHECK: stablehlo.while

// TIGHT-LABEL: @small_raised
// TIGHT: stablehlo.while

// TIGHT-LABEL: @big_raised
// TIGHT: stablehlo.while

// UNBOUNDED-LABEL: @small_raised
// UNBOUNDED-NOT: stablehlo.while

// UNBOUNDED-LABEL: @big_raised
// UNBOUNDED-NOT: stablehlo.while

module {
  func.func private @big(%arg0: memref<16xf64, 1>) {
    %cst = arith.constant 5.000000e-01 : f64
    affine.parallel (%t) = (0) to (16) {
      affine.for %i = 0 to 100000 {
        %0 = affine.load %arg0[%t] : memref<16xf64, 1>
        %1 = arith.addf %0, %cst : f64
        affine.store %1, %arg0[%t] : memref<16xf64, 1>
      }
    }
    return
  }

  func.func private @small(%arg0: memref<16xf64, 1>) {
    %cst = arith.constant 5.000000e-01 : f64
    affine.parallel (%t) = (0) to (16) {
      affine.for %i = 0 to 4 {
        %0 = affine.load %arg0[%t] : memref<16xf64, 1>
        %1 = arith.addf %0, %cst : f64
        affine.store %1, %arg0[%t] : memref<16xf64, 1>
      }
    }
    return
  }
}
