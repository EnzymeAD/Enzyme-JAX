// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-stablehlo=err_if_not_fully_raised=false | FileCheck %s

module {
  func.func private @diamond(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %r = scf.execute_region -> f64 {
        %v = affine.load %in[%t] : memref<16xf64, 1>
        %c0 = arith.constant 0.0 : f64
        %ok = arith.cmpf uge, %v, %c0 : f64
        cf.cond_br %ok, ^a, ^b
      ^a:
        %s = arith.addf %v, %v : f64
        scf.yield %s : f64
      ^b:
        %m = arith.mulf %v, %v : f64
        scf.yield %m : f64
      }
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @diamond(%[[v1:.+]]: memref<16xf64, 1>, %[[v2:.+]]: memref<16xf64, 1>) {
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:    %[[v4:.+]] = scf.execute_region -> f64 {
// CHECK-NEXT:      %[[v5:.+]] = affine.load %[[v2]][%arg2] : memref<16xf64, 1>
// CHECK-NEXT:      %[[v6:.+]] = arith.cmpf uge, %[[v5]], %[[v3]] : f64
// CHECK-NEXT:      cf.cond_br %[[v6]], ^bb1, ^bb2
// CHECK-NEXT:    ^bb1:  // pred: ^bb0
// CHECK-NEXT:      %[[v7:.+]] = arith.addf %[[v5]], %[[v5]] : f64
// CHECK-NEXT:      scf.yield %[[v7]] : f64
// CHECK-NEXT:    ^bb2:  // pred: ^bb0
// CHECK-NEXT:      %[[v8:.+]] = arith.mulf %[[v5]], %[[v5]] : f64
// CHECK-NEXT:      scf.yield %[[v8]] : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.store %[[v4]], %[[v1]][%arg2] : memref<16xf64, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// -----


// A back edge revisits a block on the live path: the raise refuses it and
// the kernel is left alone.

module {
  func.func private @looped(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %r = scf.execute_region -> f64 {
        %v = affine.load %in[%t] : memref<16xf64, 1>
        cf.br ^h(%v : f64)
      ^h(%i: f64):
        %c0 = arith.constant 0.0 : f64
        %ok = arith.cmpf uge, %i, %c0 : f64
        cf.cond_br %ok, ^h(%i : f64), ^trap
      ^trap:
        llvm.unreachable
      }
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @looped(%[[v1:.+]]: memref<16xf64, 1>, %[[v2:.+]]: memref<16xf64, 1>) {
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:    %[[v4:.+]] = scf.execute_region -> f64 {
// CHECK-NEXT:      %[[v5:.+]] = affine.load %[[v2]][%arg2] : memref<16xf64, 1>
// CHECK-NEXT:      cf.br ^bb1(%[[v5]] : f64)
// CHECK-NEXT:    ^bb1(%2: f64):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:      %[[v6:.+]] = arith.cmpf uge, %2, %[[v3]] : f64
// CHECK-NEXT:      cf.cond_br %[[v6]], ^bb1(%2 : f64), ^bb2
// CHECK-NEXT:    ^bb2:  // pred: ^bb1
// CHECK-NEXT:      llvm.unreachable
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.store %[[v4]], %[[v1]][%arg2] : memref<16xf64, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }
