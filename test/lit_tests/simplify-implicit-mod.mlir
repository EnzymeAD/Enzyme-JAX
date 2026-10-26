// RUN: enzymexlamlir-opt %s --simplify-affine-exprs | FileCheck %s

// The store index %i * 16 + %j - (%i floordiv 3) * 48 is an implicit
// remainder: it equals (%i mod 3) * 16 + %j and should be folded to it.

// CHECK-LABEL: func.func private @implicit_mod_fold
// CHECK: affine.parallel (%[[I:.+]], %[[J:.+]]) = (0, 0) to (90, 16) {
// CHECK-NEXT: %[[C:.+]] = arith.constant 1 : i64
// CHECK-NEXT: affine.store %[[C]], %{{.*}}[%[[I]] floordiv 9, %[[J]] + (%[[I]] mod 3) * 16] : memref<20x50xi64, 1>

module {
  func.func private @implicit_mod_fold(%arg0: memref<20x50xi64, 1>) {
    affine.parallel (%i, %j) = (0, 0) to (90, 16) {
      %c = arith.constant 1 : i64
      affine.store %c, %arg0[%i floordiv 9, %i * 16 + %j - (%i floordiv 3) * 48] : memref<20x50xi64, 1>
    }
    return
  }
}
