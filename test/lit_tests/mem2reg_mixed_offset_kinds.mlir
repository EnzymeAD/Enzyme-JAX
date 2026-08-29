// RUN: enzymexlamlir-opt %s --polygeist-mem2reg | FileCheck %s

// A store at a loop-carried index writes some slot every iteration; which one
// is exactly what is not known. Comparing that index against a constant slot
// has no answer but "maybe", and answering anything else forwarded the value
// stored before the loop across the loop that overwrote it: MFEM's
// Values3D quadrature kernel read its pre-loop scratch values instead of the
// interpolated ones, every partial-assembly operator disagreed with full
// assembly, and iterative solvers span forever on the mismatch.

module {
  func.func @no_forward_across_symbolic_store(%v: f64, %w: f64) -> f64 {
    %a = memref.alloca() : memref<16xf64>
    %z = arith.constant 1 : index
    affine.store %v, %a[1] : memref<16xf64>
    affine.parallel (%i) = (0) to (4) {
      affine.store %w, %a[%i] : memref<16xf64>
    }
    %r = affine.load %a[1] : memref<16xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @no_forward_across_symbolic_store
// CHECK:         affine.parallel
// CHECK:         %[[R:.+]] = affine.load
// CHECK:         return %[[R]]
