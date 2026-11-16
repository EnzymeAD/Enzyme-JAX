// RUN: enzymexlamlir-opt %s --simplify-affine-exprs | FileCheck %s

// A symbolic bound on an enclosing loop puts a symbol in the domain; the
// accesses below it still fold against the constant bounds of their own loop.
func.func @symbolic_grid(%n: index, %v: f64) {
  %alloca = memref.alloca() : memref<4x3x3xf64>
  affine.parallel (%k) = (0) to (symbol(%n)) {
    affine.parallel (%i, %j) = (0, 0) to (3, 3) {
      affine.store %v, %alloca[(%j + %i floordiv 3) floordiv 3, (%j + %i floordiv 3) mod 3, %i mod 3] : memref<4x3x3xf64>
    }
  }
  return
}

// The access has a symbol operand of its own that no enclosing loop bounds.
func.func @symbol_operand(%n: index, %s: index, %v: f64, %m: memref<?xf64>) {
  affine.for %k = 0 to %n {
    affine.parallel (%i) = (0) to (3) {
      affine.store %v, %m[%s * 3 + %i floordiv 3 + %k] : memref<?xf64>
    }
  }
  return
}

// A condition under a symbolic bound, mixing the loop's dim with a symbol.
func.func @condition_under_symbolic_bound(%n: index, %s: index, %v: f64, %m: memref<?xf64>) {
  affine.parallel (%k) = (0) to (symbol(%n)) {
    affine.parallel (%i) = (0) to (3) {
      affine.if affine_set<(d0, d1)[s0] : (d0 floordiv 3 + s0 - d1 >= 0)>(%i, %k)[%s] {
        affine.store %v, %m[%k] : memref<?xf64>
      }
    }
  }
  return
}

// A single-trip inner loop pins its iv to the outer one; the domain equality
// lets the access read the outer iv instead, which becomes a new operand.
func.func @single_trip(%v: f64, %m: memref<?xf64>) {
  affine.for %i = 0 to 8 {
    affine.for %j = affine_map<(d0) -> (d0)>(%i) to affine_map<(d0) -> (d0 + 1)>(%i) {
      affine.store %v, %m[%j] : memref<?xf64>
      affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%j, %i) {
        affine.store %v, %m[%j - %i] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  #set = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
// CHECK-NEXT:#set1 = affine_set<() : (0 >= 0)>
// CHECK-NEXT:module {
// CHECK-NEXT:  func.func @symbolic_grid(%arg0: index, %arg1: f64) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<4x3x3xf64>
// CHECK-NEXT:    affine.parallel (%arg2) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:      affine.parallel (%arg3, %arg4) = (0, 0) to (3, 3) {
// CHECK-NEXT:        affine.store %arg1, %alloca[0, %arg4, %arg3] : memref<4x3x3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @symbol_operand(%arg0: index, %arg1: index, %arg2: f64, %arg3: memref<?xf64>) {
// CHECK-NEXT:    affine.for %arg4 = 0 to %arg0 {
// CHECK-NEXT:      affine.parallel (%arg5) = (0) to (3) {
// CHECK-NEXT:        affine.store %arg2, %arg3[%arg4 + %arg1 * 3] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @condition_under_symbolic_bound(%arg0: index, %arg1: index, %arg2: f64, %arg3: memref<?xf64>) {
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:      affine.parallel (%arg5) = (0) to (3) {
// CHECK-NEXT:        affine.if #set(%arg4)[%arg1] {
// CHECK-NEXT:          affine.store %arg2, %arg3[%arg4] : memref<?xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @single_trip(%arg0: f64, %arg1: memref<?xf64>) {
// CHECK-NEXT:    affine.for %arg2 = 0 to 8 {
// CHECK-NEXT:      affine.store %arg0, %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:      affine.if #set1() {
// CHECK-NEXT:        affine.store %arg0, %arg1[0] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
