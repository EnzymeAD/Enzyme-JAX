// RUN: enzymexlamlir-opt %s --test-polymer 2>&1 | FileCheck %s

module {
  func.func @affine(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      %alloc = memref.alloc() : memref<4xf32>
      affine.parallel (%arg5) = (0) to (4) {
        %1 = affine.load %arg1[%arg5] : memref<?xf32>
        affine.store %1, %alloc[%arg5] : memref<4xf32>
        %2 = arith.mulf %1, %arg0 : f32
        affine.store %2, %arg3[%arg5] : memref<?xf32>
      }
      memref.dealloc %alloc : memref<4xf32>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK: Processing enzymexla.gpu_wrapper
// CHECK: Schedule:
// CHECK: domain: "{ S5_affine_yield[i0] : 0 <= i0 <= 3; S2_affine_store[i0] : 0 <= i0 <= 3; S1_affine_load[i0] : 0 <= i0 <= 3; S3_arith_mulf[i0] : 0 <= i0 <= 3; S4_affine_store[i0] : 0 <= i0 <= 3; S0_memref_alloc[]; S6_memref_dealloc[] }"
// CHECK: child:
// CHECK:   sequence:
// CHECK:   - filter: "{ S0_memref_alloc[] }"
// CHECK:   - filter: "{ S1_affine_load[i0]; S4_affine_store[i0]; S2_affine_store[i0]; S5_affine_yield[i0]; S3_arith_mulf[i0] }"
// CHECK:     child:
// CHECK:       schedule: "L0_affine_parallel[{ S5_affine_yield[i0] -> [(i0)]; S2_affine_store[i0] -> [(i0)]; S1_affine_load[i0] -> [(i0)]; S3_arith_mulf[i0] -> [(i0)]; S4_affine_store[i0] -> [(i0)] }]"
// CHECK:       permutable: 1
// CHECK:       child:
// CHECK:         sequence:
// CHECK:         - filter: "{ S1_affine_load[i0] }"
// CHECK:         - filter: "{ S2_affine_store[i0] }"
// CHECK:         - filter: "{ S3_arith_mulf[i0] }"
// CHECK:         - filter: "{ S4_affine_store[i0] }"
// CHECK:         - filter: "{ S5_affine_yield[i0] }"
// CHECK:   - filter: "{ S6_memref_dealloc[] }"
// CHECK: Accesses:
// CHECK: domain: "{ S5_affine_yield[i0] : 0 <= i0 <= 3; S2_affine_store[i0] : 0 <= i0 <= 3; S1_affine_load[i0] : 0 <= i0 <= 3; S3_arith_mulf[i0] : 0 <= i0 <= 3; S4_affine_store[i0] : 0 <= i0 <= 3; S0_memref_alloc[]; S6_memref_dealloc[] }"
// CHECK: accesses:
// CHECK:   - S0_memref_alloc:
// CHECK:   - S1_affine_load:
// CHECK:         - read "{ S1_affine_load[i0] -> A_func_func_arg_1_0[i0] }"
// CHECK:         - must_write "{ S1_affine_load[i0] -> A_affine_load_res_1[] }"
// CHECK:   - S2_affine_store:
// CHECK:         - must_write "{ S2_affine_store[i0] -> A_memref_alloc_res_2[i0] }"
// CHECK:         - read "{ S2_affine_store[i0] -> A_affine_load_res_1[] }"
// CHECK:   - S3_arith_mulf:
// CHECK:         - must_write "{ S3_arith_mulf[i0] -> A_arith_mulf_res_3[] }"
// CHECK:         - read "{ S3_arith_mulf[i0] -> A_affine_load_res_1[] }"
// CHECK:         - read "{ S3_arith_mulf[i0] -> A_func_func_arg_0_4[] }"
// CHECK:   - S4_affine_store:
// CHECK:         - must_write "{ S4_affine_store[i0] -> A_func_func_arg_3_5[i0] }"
// CHECK:         - read "{ S4_affine_store[i0] -> A_arith_mulf_res_3[] }"
// CHECK:   - S5_affine_yield:
// CHECK:         - kill "{ S5_affine_yield[i0] -> A_affine_load_res_1[] : 0 <= i0 <= 3 }"
// CHECK:         - kill "{ S5_affine_yield[i0] -> A_arith_mulf_res_3[] : 0 <= i0 <= 3 }"
// CHECK:   - S6_memref_dealloc:
// CHECK:   - S7_enzymexla_polygeist_yield:
// CHECK: Processing func.func
// CHECK: Failed to build scop
