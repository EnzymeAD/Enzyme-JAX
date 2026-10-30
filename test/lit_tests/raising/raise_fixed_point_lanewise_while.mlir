// RUN: enzymexlamlir-opt %s -polygeist-mem2reg --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A per-point Newton iteration for sqrt(a) inside a kernel, as Breeze's
// NewtonSolver loops are: each lane has its own convergence test. Marked as a
// fixed point, the loop raises to one stablehlo.while over the grid that keeps
// going while any lane has not converged. It also carries the lanes still
// active (named by enzyme.fixed_point_mask), and only those take the step.
// (Unmarked, a lane-wise predicate cannot be raised at all: the raising
// reports an error for the kernel, which is what the stock pipeline avoids by
// converting the loop to a counted one and unrolling it.)
func.func @newton(%u: memref<100xf64, 1>, %a: memref<100xf64, 1>, %tol: memref<f64, 1>) {
  affine.parallel (%t) = (0) to (100) {
    %x0 = affine.load %u[%t] : memref<100xf64, 1>
    %av = affine.load %a[%t] : memref<100xf64, 1>
    %tl = affine.load %tol[] : memref<f64, 1>
    %r = scf.while (%x = %x0) : (f64) -> f64 {
      %x2 = arith.mulf %x, %x : f64
      %res = arith.subf %x2, %av : f64
      %go = arith.cmpf ogt, %res, %tl : f64
      scf.condition(%go) %x : f64
    } do {
    ^bb0(%x: f64):
      %x2 = arith.mulf %x, %x : f64
      %r0 = arith.subf %x2, %av : f64
      %two = arith.constant 2.0 : f64
      %d = arith.mulf %two, %x : f64
      %q = arith.divf %r0, %d : f64
      %xn = arith.subf %x, %q : f64
      scf.yield %xn : f64
    } attributes {enzyme.fixed_point}
    affine.store %r, %u[%t] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @newton_raised
// CHECK:       stablehlo.reduce
// CHECK-SAME:    applies stablehlo.or
// CHECK:       stablehlo.while(%[[GO:[^ ]+]] = %{{[^,]+}}, %[[ACTIVE:[^ ]+]] = %{{[^,]+}}, %[[X:[^ ]+]] = %{{[^,]+}},
// CHECK-SAME:    attributes {enzyme.fixed_point, enzyme.fixed_point_mask = 1 : i64}
// CHECK:       cond {
// CHECK:       } do {
// CHECK:         %[[STEP:.+]] = arith.subf %[[X]], %{{.+}} : tensor<100xf64>
// CHECK:         %[[STILL:.+]] = stablehlo.and %[[ACTIVE]], %{{.+}} : tensor<100xi1>
// CHECK:         %[[ANY:.+]] = stablehlo.reduce(%[[STILL]]
// CHECK-SAME:      applies stablehlo.or
// CHECK:         %[[NEXT:.+]] = stablehlo.select %[[ACTIVE]], %[[STEP]], %[[X]]
// CHECK:         stablehlo.return %[[ANY]], %[[STILL]], %[[NEXT]],
