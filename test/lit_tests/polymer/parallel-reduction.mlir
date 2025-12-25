// RUN: enzymexlamlir-opt %s --test-polymer 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @affine2(%arg1: memref<4xf32>) -> f32 {
    %0 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
      %1 = affine.load %arg1[%arg5] : memref<4xf32>
      affine.yield %1 : f32
    }
    return %0 : f32
  }
}

// CHECK: Processing func.func
// CHECK: Schedule:
// CHECK: domain: "{ S1_affine_load[i0] : 0 <= i0 <= 3; S2_affine_yield[i0] : 0 <= i0 <= 3; S0_enzymexla_store_var[] }"
// CHECK: child:
// CHECK:   sequence:
// CHECK:   - filter: "{ S0_enzymexla_store_var[] }"
// CHECK:   - filter: "{ S1_affine_load[i0]; S2_affine_yield[i0] }"
// CHECK:     child:
// CHECK:       schedule: "L0_affine_parallel[{ S1_affine_load[i0] -> [(i0)]; S2_affine_yield[i0] -> [(i0)] }]"
// CHECK:       permutable: 1
// CHECK:       child:
// CHECK:         sequence:
// CHECK:         - filter: "{ S1_affine_load[i0] }"
// CHECK:         - filter: "{ S2_affine_yield[i0] }"
// CHECK: Accesses:
// CHECK: domain: "{ S1_affine_load[i0] : 0 <= i0 <= 3; S2_affine_yield[i0] : 0 <= i0 <= 3; S0_enzymexla_store_var[] }"
// CHECK: accesses:
// CHECK:   - S0_enzymexla_store_var:
// CHECK:         - must_write "{ S0_enzymexla_store_var[] -> A_affine_parallel_res_0[] }"
// CHECK:   - S1_affine_load:
// CHECK:         - read "{ S1_affine_load[i0] -> A_func_func_arg_0_1[i0] }"
// CHECK:         - must_write "{ S1_affine_load[i0] -> A_affine_load_res_2[] }"
// CHECK:   - S2_affine_yield:
// CHECK:         - must_write "{ S2_affine_yield[i0] -> A_affine_parallel_res_0[] }"
// CHECK:         - read "{ S2_affine_yield[i0] -> A_affine_load_res_2[] }"
// CHECK:         - kill "{ S2_affine_yield[i0] -> A_affine_load_res_2[] : 0 <= i0 <= 3 }"
// CHECK:   - S3_func_return:
