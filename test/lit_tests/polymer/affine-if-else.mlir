// RUN: enzymexlamlir-opt %s --test-polymer 2>&1 | FileCheck %s

#interior = affine_set<(d0) : (d0 - 2 >= 0, -d0 + 5 >= 0)>
#first = affine_set<(d0) : (d0 == 0)>

func.func @branches(%out: memref<8xi32>, %a: i32, %b: i32, %c: i32) {
  affine.parallel (%i) = (0) to (8) {
    affine.if #interior(%i) {
      affine.store %a, %out[%i] : memref<8xi32>
    } else {
      affine.if #first(%i) {
        affine.store %b, %out[%i] : memref<8xi32>
      } else {
        affine.store %c, %out[%i] : memref<8xi32>
      }
    }
  }
  return
}

// CHECK: Schedule:
// CHECK-DAG: S0_affine_store[i0] : 2 <= i0 <= 5
// CHECK-DAG: S2_affine_store[0]
// CHECK-DAG: S4_affine_store[1]
// CHECK-DAG: S4_affine_store[i0] : 6 <= i0 <= 7
// CHECK: child:
