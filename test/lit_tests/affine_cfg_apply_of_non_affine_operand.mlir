// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// An affine.apply may take any index for operand, valid affine dimension or
// symbol or not. Raising a memref.load whose index it computes composes the
// apply into the access map, which is only legal when its operands are valid
// for the access: j, loaded inside the affine.parallel, is neither, and the
// load must stay a memref.load -- as it does when the same 2 * j + 1 is
// spelled in arith.

#re = affine_map<(d0) -> (d0 * 2)>
#im = affine_map<(d0) -> (d0 * 2 + 1)>

func.func private @apply_of_loaded_index(%out: memref<8xf64, 1>, %in: memref<16xf64, 1>, %perm: memref<8xi64, 1>) {
  affine.parallel (%i) = (0) to (8) {
    %j64 = affine.load %perm[%i] : memref<8xi64, 1>
    %j = arith.index_cast %j64 : i64 to index
    %a = affine.apply #re(%j)
    %b = affine.apply #im(%j)
    %x = memref.load %in[%a] : memref<16xf64, 1>
    %y = memref.load %in[%b] : memref<16xf64, 1>
    %s = arith.addf %x, %y : f64
    affine.store %s, %out[%i] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @apply_of_loaded_index
// CHECK-SAME:    (%{{.+}}: memref<8xf64, 1>, %[[IN:.+]]: memref<16xf64, 1>, %{{.+}}: memref<8xi64, 1>)
// CHECK-NOT:     affine.load %[[IN]]
// CHECK:         memref.load %[[IN]]
// CHECK:         memref.load %[[IN]]
// CHECK-NOT:     affine.load %[[IN]]

// An affine.apply of the loop's own index still composes into the access.
func.func private @apply_of_loop_index(%out: memref<8xf64, 1>, %in: memref<16xf64, 1>) {
  affine.parallel (%i) = (0) to (8) {
    %a = affine.apply #re(%i)
    %b = affine.apply #im(%i)
    %x = memref.load %in[%a] : memref<16xf64, 1>
    %y = memref.load %in[%b] : memref<16xf64, 1>
    %s = arith.addf %x, %y : f64
    affine.store %s, %out[%i] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @apply_of_loop_index
// CHECK-SAME:    (%{{.+}}: memref<8xf64, 1>, %[[IN:.+]]: memref<16xf64, 1>)
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (8) {
// CHECK-DAG:       affine.load %[[IN]][%[[I]] * 2] : memref<16xf64, 1>
// CHECK-DAG:       affine.load %[[IN]][%[[I]] * 2 + 1] : memref<16xf64, 1>
// CHECK-NOT:     memref.load %[[IN]]
