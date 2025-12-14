// RUN: enzymexlamlir-opt %s --test-polymer 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @affine2(%arg0: f32, %arg2: memref<?xf32>) {
    affine.parallel (%arg5) = (0) to (4) {
      %5 = enzyme.affine_atomic_rmw addf %arg0, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
      affine.yield
    }
    return
  }
}

// CHECK: Processing func.func
// CHECK: Schedule:
// CHECK: domain: "{ S0_enzyme_affine_atomic_rmw[i0] : 0 <= i0 <= 3; S1_affine_yield[i0] : 0 <= i0 <= 3 }"
// CHECK: child:
// CHECK:   schedule: "L0_affine_parallel[{ S0_enzyme_affine_atomic_rmw[i0] -> [(i0)]; S1_affine_yield[i0] -> [(i0)] }]"
// CHECK:   permutable: 1
// CHECK:   child:
// CHECK:     sequence:
// CHECK:     - filter: "{ S0_enzyme_affine_atomic_rmw[i0] }"
// CHECK:     - filter: "{ S1_affine_yield[i0] }"
// CHECK: Accesses:
// CHECK: domain: "{ S0_enzyme_affine_atomic_rmw[i0] : 0 <= i0 <= 3; S1_affine_yield[i0] : 0 <= i0 <= 3 }"
// CHECK: accesses:
// CHECK:   - S0_enzyme_affine_atomic_rmw:
// CHECK:         - read "{ S0_enzyme_affine_atomic_rmw[i0] -> A_func_func_arg_0_0[] }"
// CHECK:         - read "{ S0_enzyme_affine_atomic_rmw[i0] -> A_func_func_arg_1_1[i0] }"
// CHECK:         - must_write "{ S0_enzyme_affine_atomic_rmw[i0] -> A_func_func_arg_1_1[i0] }"
// CHECK:         - must_write "{ S0_enzyme_affine_atomic_rmw[i0] -> A_enzyme_affine_atomic_rmw_res_2[] }"
// CHECK:   - S1_affine_yield:
// CHECK:         - kill "{ S1_affine_yield[i0] -> A_enzyme_affine_atomic_rmw_res_2[] : 0 <= i0 <= 3 }"
// CHECK:   - S2_func_return:
