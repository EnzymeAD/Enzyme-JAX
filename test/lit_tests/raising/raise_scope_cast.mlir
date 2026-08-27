// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A shape-erasing cast of static scratch only renames the buffer; accesses
// go straight to the static source.
func.func @castuse(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
  %scr = memref.alloca() : memref<16xf64>
  %dyn = memref.cast %scr : memref<16xf64> to memref<?xf64>
  affine.parallel (%t) = (0) to (16) {
    %v = affine.load %in[%t] : memref<16xf64, 1>
    affine.store %v, %dyn[%t] : memref<?xf64>
    %r = affine.load %dyn[15 - %t] : memref<?xf64>
    affine.store %r, %out[%t] : memref<16xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @castuse_raised(
// CHECK-NOT: memref.cast
// CHECK: stablehlo.reverse

// -----

// An inliner-wrapped callee arrives as an execute_region whose CFG carries a
// trap arm; the branch takes its one live successor, the straight line
// merges, and the scope dissolves into the surrounding kernel.
func.func @scoped(%out: memref<16xi32, 1>, %in: memref<16xi32, 1>) {
  affine.parallel (%t) = (0) to (16) {
    %v = affine.load %in[%t] : memref<16xi32, 1>
    %r = scf.execute_region -> i32 {
      %c100 = arith.constant 100 : i32
      %c = arith.cmpi slt, %v, %c100 : i32
      cf.cond_br %c, ^bb1, ^bb2
    ^bb1:
      %c1 = arith.constant 1 : i32
      %s = arith.addi %v, %c1 : i32
      scf.yield %s : i32
    ^bb2:
      llvm.unreachable
    }
    affine.store %r, %out[%t] : memref<16xi32, 1>
  }
  return
}

// CHECK-LABEL: func.func private @scoped_raised(
// CHECK-NOT: scf.execute_region
// CHECK: stablehlo.add
