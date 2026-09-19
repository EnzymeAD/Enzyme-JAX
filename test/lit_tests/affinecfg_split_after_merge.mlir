// RUN: enzymexlamlir-opt --affine-cfg="enable_split_on_affine_if_constants=true" %s | FileCheck %s

// A 2-D launch whose rows are `%i + %j * 16`, with halo guards over that
// expression (EnzymeAD/Enzyme-JAX#3202). The inductions merge before the
// split copies the body under each guard: a copy specialized to the halo
// rows (`%i + 8`) no longer indexes with the uniform expression the merge
// needs, so the split must come second.

module {
  func.func @split_blocks_merge(%m: memref<1536x3072xf32, 1>, %a: f32, %b: f32) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (16, 95, 3056) {
      // The halo masks: `%i + %j * 16` in [0, 2] and <= 1517.
      %lo = affine.if affine_set<(d0, d1) : (d0 + d1 * 16 + 1 >= 0, -d0 - d1 * 16 + 2 >= 0)>(%i, %j) -> i1 {
        affine.yield %false : i1
      } else {
        affine.yield %true : i1
      }
      %hi = affine.if affine_set<(d0, d1) : (-d0 - d1 * 16 + 1517 >= 0)>(%i, %j) -> i1 {
        affine.yield %true : i1
      } else {
        affine.yield %false : i1
      }
      %interior = arith.andi %lo, %hi : i1
      %sel = arith.select %interior, %a, %b : f32
      %c = arith.cmpf olt, %cst, %sel : f32
      %r = scf.if %c -> f32 {
        %l = affine.load %m[%i + %j * 16 + 8, %k + 8] : memref<1536x3072xf32, 1>
        scf.yield %l : f32
      } else {
        %l = affine.load %m[%i + %j * 16 + 9, %k + 8] : memref<1536x3072xf32, 1>
        scf.yield %l : f32
      }
      // The bulk of the body: these are the accesses that lose their slice when
      // the pair fails to merge.
      %p = affine.load %m[%i + %j * 16 + 7, %k + 8] : memref<1536x3072xf32, 1>
      %q = arith.addf %r, %p : f32
      affine.store %q, %m[%i + %j * 16 + 8, %k + 8] : memref<1536x3072xf32, 1>
    }
    return
  }
}

// CHECK:  #set = affine_set<(d0) : (-d0 + 1517 >= 0)>
// CHECK-NEXT:  #set1 = affine_set<(d0) : (d0 + 1 >= 0, -d0 + 2 >= 0)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @split_blocks_merge(%arg0: memref<1536x3072xf32, 1>, %arg1: f32, %arg2: f32) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      affine.parallel (%arg3, %arg4) = (0, 0) to (1520, 3056) {
// CHECK-NEXT:        %0 = affine.if #set(%arg3) -> f32 {
// CHECK-NEXT:          %3 = affine.if #set1(%arg3) -> f32 {
// CHECK-NEXT:            %4 = arith.cmpf olt, %cst, %arg2 : f32
// CHECK-NEXT:            %5 = scf.if %4 -> (f32) {
// CHECK-NEXT:              %6 = affine.load %arg0[%arg3 + 8, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:              scf.yield %6 : f32
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %6 = affine.load %arg0[%arg3 + 9, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:              scf.yield %6 : f32
// CHECK-NEXT:            }
// CHECK-NEXT:            affine.yield %5 : f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %4 = arith.cmpf olt, %cst, %arg1 : f32
// CHECK-NEXT:            %5 = scf.if %4 -> (f32) {
// CHECK-NEXT:              %6 = affine.load %arg0[%arg3 + 8, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:              scf.yield %6 : f32
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %6 = affine.load %arg0[%arg3 + 9, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:              scf.yield %6 : f32
// CHECK-NEXT:            }
// CHECK-NEXT:            affine.yield %5 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.yield %3 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %3 = arith.cmpf olt, %cst, %arg2 : f32
// CHECK-NEXT:          %4 = scf.if %3 -> (f32) {
// CHECK-NEXT:            %5 = affine.load %arg0[%arg3 + 8, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:            scf.yield %5 : f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %5 = affine.load %arg0[%arg3 + 9, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:            scf.yield %5 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.yield %4 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %1 = affine.load %arg0[%arg3 + 7, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:        %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:        affine.store %2, %arg0[%arg3 + 8, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
