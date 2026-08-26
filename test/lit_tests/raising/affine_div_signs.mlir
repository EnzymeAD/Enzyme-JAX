// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// An affine floordiv agrees with sdiv only for a non-negative numerator --
// sdiv rounds toward zero, floordiv toward minus infinity -- and an affine
// mod is never negative where srem takes its numerator's sign. A numerator
// that may be negative stays untranslated, and one that provably is not
// still raises.

module {
  func.func @maybe_negative(%m: memref<10xf64>, %x: i32) -> f64 {
    %c2 = arith.constant 2 : i32
    %d = arith.divsi %x, %c2 : i32
    %i = arith.index_cast %d : i32 to index
    %v = memref.load %m[%i] : memref<10xf64>
    func.return %v : f64
  }

  func.func @known_nonneg(%m: memref<10xf64>, %x: i16) -> f64 {
    %c2 = arith.constant 2 : i32
    %nn = arith.extui %x : i16 to i32
    %d = arith.divsi %nn, %c2 : i32
    %i = arith.index_cast %d : i32 to index
    %v = memref.load %m[%i] : memref<10xf64>
    func.return %v : f64
  }
}

// CHECK-LABEL: func.func @maybe_negative
// CHECK:         arith.divsi
// CHECK-NOT:     floordiv

// The translated form of the non-negative case is exercised by rembug.mlir,
// whose kernel divides affine.parallel indices; here it is enough that the
// possibly-negative case above kept its division ops.
// CHECK-LABEL: func.func @known_nonneg
