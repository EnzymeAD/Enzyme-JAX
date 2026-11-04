// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-triton | FileCheck %s
// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-triton --canonicalize --lower-triton | FileCheck %s --check-prefix=LOWER

// 1. 1D power-of-two pointwise kernel via enzymexla.jit_call
module {
  func.func private @vec_add(%a: memref<256xf32, 1>, %b: memref<256xf32, 1>, %out: memref<256xf32, 1>) {
    affine.parallel (%i) = (0) to (256) {
      %va = affine.load %a[%i] : memref<256xf32, 1>
      %vb = affine.load %b[%i] : memref<256xf32, 1>
      %sum = arith.addf %va, %vb : f32
      affine.store %sum, %out[%i] : memref<256xf32, 1>
    }
    return
  }

  func.func @main(%ta: tensor<256xf32>, %tb: tensor<256xf32>, %tout: tensor<256xf32>) -> tensor<256xf32> {
    %res = enzymexla.jit_call @vec_add(%ta, %tb, %tout) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %res : tensor<256xf32>
  }
}

// CHECK-LABEL: enzymexla_tt_ext.module @vec_add_tt
// CHECK: builtin.module @vec_add_inner attributes {enzymexla.num_stages = 3 : i32, enzymexla.num_warps = 4 : i32}
// CHECK: tt.func @vec_add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>)
// CHECK: tt.get_program_id x : i32
// CHECK: tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK: tt.load
// CHECK: tt.load
// CHECK: arith.addf {{.*}} : tensor<128xf32>
// CHECK: tt.store
// CHECK: tt.return
// CHECK-LABEL: func.func @main
// CHECK: enzymexla_tt_ext.call @vec_add_tt::@vec_add_inner::@vec_add_kernel
// LOWER-LABEL: func.func @main
// LOWER: stablehlo.custom_call @__gpu$xla.gpu.triton(%arg0, %arg1, %arg2)
// LOWER-SAME: grid_x = 2 : i32, grid_y = 1 : i32, grid_z = 1 : i32

// -----

// 2. Non-power-of-two 2D masked kernel with affine.for reduction + affine.if predication
#set_diag = affine_set<(d0, d1) : (d0 - d1 >= 0)>
module {
  func.func private @mat_row_scale(%A: memref<20x36xf32, 1>, %B: memref<36x12xf32, 1>, %C: memref<20x12xf32, 1>) {
    %zero = arith.constant 0.0 : f32
    affine.parallel (%i, %j) = (0, 0) to (20, 12) {
      %acc = affine.for %k = 0 to 36 iter_args(%sum = %zero) -> (f32) {
        %a = affine.load %A[%i, %k] : memref<20x36xf32, 1>
        %b = affine.load %B[%k, %j] : memref<36x12xf32, 1>
        %prod = arith.mulf %a, %b : f32
        %next = arith.addf %sum, %prod : f32
        affine.yield %next : f32
      }
      %outv = affine.if #set_diag(%i, %j) -> f32 {
        %t = math.tanh %acc : f32
        affine.yield %t : f32
      } else {
        affine.yield %acc : f32
      }
      affine.store %outv, %C[%i, %j] : memref<20x12xf32, 1>
    }
    return
  }

  func.func @entry_2d(%A: tensor<20x36xf32>, %B: tensor<36x12xf32>, %C: tensor<20x12xf32>) -> tensor<20x12xf32> {
    %r = enzymexla.jit_call @mat_row_scale(%A, %B, %C) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> : (tensor<20x36xf32>, tensor<36x12xf32>, tensor<20x12xf32>) -> tensor<20x12xf32>
    return %r : tensor<20x12xf32>
  }
}

// CHECK-LABEL: enzymexla_tt_ext.module @mat_row_scale_tt
// CHECK: tt.func @mat_row_scale_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>)
// CHECK: scf.for
// CHECK: arith.mulf {{.*}} : tensor<32x16xf32>
// CHECK: arith.addf {{.*}} : tensor<32x16xf32>
// CHECK: tt.store
// CHECK-LABEL: func.func @entry_2d
// CHECK: enzymexla_tt_ext.call @mat_row_scale_tt::@mat_row_scale_inner::@mat_row_scale_kernel
// LOWER-LABEL: func.func @entry_2d
// LOWER: stablehlo.custom_call @__gpu$xla.gpu.triton

// -----

// 3. Standalone memref function + enzyme.affine_atomic_rmw
module {
  func.func private @atomic_scatter(%buf: memref<8xf32, 1>, %idx: memref<32xi32, 1>, %val: memref<32xf32, 1>) {
    affine.parallel (%t) = (0) to (32) {
      %i = affine.load %idx[%t] : memref<32xi32, 1>
      %iidx = arith.index_cast %i : i32 to index
      %v = affine.load %val[%t] : memref<32xf32, 1>
      %old = enzyme.affine_atomic_rmw addf %v, %buf, (affine_map<(d0) -> (d0)>)[%iidx] monotonic : (f32, memref<8xf32, 1>) -> f32
    }
    return
  }
}

// CHECK-LABEL: enzymexla_tt_ext.module @atomic_scatter_tt
// CHECK: tt.func @atomic_scatter_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>)
// CHECK: tt.atomic_rmw fadd, acq_rel, gpu
// CHECK-LABEL: func.func private @atomic_scatter_raised(%arg0: tensor<8xf32>, %arg1: tensor<32xi32>, %arg2: tensor<32xf32>) -> (tensor<8xf32>, tensor<32xi32>, tensor<32xf32>)
// CHECK: %[[UPDATED:.+]] = enzymexla_tt_ext.call @atomic_scatter_tt::@atomic_scatter_inner::@atomic_scatter_kernel
// CHECK: return %[[UPDATED]], %arg1, %arg2
// LOWER-LABEL: func.func private @atomic_scatter_raised
// LOWER: stablehlo.custom_call @__gpu$xla.gpu.triton

// -----

// 4. Dynamic parallel bound read from rank-0 scalar memref (grid-stride loop)
module {
  func.func private @dyn_scale(%x: memref<512xf32, 1>, %n_ref: memref<i64, 1>) {
    %n64 = affine.load %n_ref[] : memref<i64, 1>
    %n = arith.index_cast %n64 : i64 to index
    %two = arith.constant 2.0 : f32
    affine.parallel (%i) = (0) to (%n) {
      %v = affine.load %x[%i] : memref<512xf32, 1>
      %scaled = arith.mulf %v, %two : f32
      affine.store %scaled, %x[%i] : memref<512xf32, 1>
    }
    return
  }

  func.func @run_dyn(%x: tensor<512xf32>, %n: tensor<i64>) -> tensor<512xf32> {
    %r = enzymexla.jit_call @dyn_scale(%x, %n) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<512xf32>, tensor<i64>) -> tensor<512xf32>
    return %r : tensor<512xf32>
  }
}

// CHECK-LABEL: enzymexla_tt_ext.module @dyn_scale_tt
// CHECK: tt.func @dyn_scale_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>)
// CHECK: scf.for
// CHECK: tt.load
// CHECK: tt.store
// CHECK-LABEL: func.func @run_dyn
// CHECK: enzymexla_tt_ext.call @dyn_scale_tt::@dyn_scale_inner::@dyn_scale_kernel
// LOWER-LABEL: func.func @run_dyn
// LOWER: stablehlo.custom_call @__gpu$xla.gpu.triton

// -----

// 5. enzymexla.jit_region and local alloca scratch promotion
module {
  func.func @region_kernel(%in: tensor<64xf32>, %out: tensor<64xf32>) -> tensor<64xf32> {
    %res = "enzymexla.jit_region"(%in, %out) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> ({
    ^bb0(%m_in: memref<64xf32>, %m_out: memref<64xf32>):
      affine.parallel (%i) = (0) to (64) {
        %tmp = memref.alloca() : memref<f32>
        %v = affine.load %m_in[%i] : memref<64xf32>
        %sq = arith.mulf %v, %v : f32
        affine.store %sq, %tmp[] : memref<f32>
        %loaded = affine.load %tmp[] : memref<f32>
        affine.store %loaded, %m_out[%i] : memref<64xf32>
      }
      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %res : tensor<64xf32>
  }
}

// CHECK-LABEL: enzymexla_tt_ext.module @jit_region_0_tt
// CHECK: tt.func @jit_region_0_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>)
// CHECK-NOT: memref.alloca
// CHECK: arith.mulf {{.*}} : tensor<64xf32>
// CHECK-LABEL: func.func @region_kernel
// CHECK: enzymexla_tt_ext.call @jit_region_0_tt::@jit_region_0_inner::@jit_region_0_kernel
// LOWER-LABEL: func.func @region_kernel
// LOWER: stablehlo.custom_call @__gpu$xla.gpu.triton

// -----

// 6. Sequential multi-launch segmentation over dynamic-outer-dim memref + gpu_wrapper
module {
  func.func private @two_stage(%a: memref<?x16xf32, 1>, %b: memref<?x16xf32, 1>) {
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    affine.parallel (%i, %j) = (0, 0) to (4, 16) {
      %va = affine.load %a[%i, %j] : memref<?x16xf32, 1>
      %s1 = arith.addf %va, %one : f32
      affine.store %s1, %b[%i, %j] : memref<?x16xf32, 1>
    }
    affine.parallel (%i, %j) = (0, 0) to (4, 16) {
      %vb = affine.load %b[%i, %j] : memref<?x16xf32, 1>
      %s2 = arith.mulf %vb, %two : f32
      affine.store %s2, %b[%i, %j] : memref<?x16xf32, 1>
    }
    return
  }

  func.func @run_two_stage(%a: tensor<4x16xf32>, %b: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %r = enzymexla.jit_call @two_stage(%a, %b) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    return %r : tensor<4x16xf32>
  }
}

// CHECK-DAG: enzymexla_tt_ext.module @two_stage_0_tt
// CHECK-DAG: enzymexla_tt_ext.module @two_stage_1_tt
// CHECK-LABEL: func.func @run_two_stage
// CHECK: %[[S1:.+]] = enzymexla_tt_ext.call @two_stage_0_tt::@two_stage_0_inner::@two_stage_0_kernel {{.*}}(%arg0, %arg1)
// CHECK: %[[S2:.+]] = enzymexla_tt_ext.call @two_stage_1_tt::@two_stage_1_inner::@two_stage_1_kernel {{.*}}(%[[S1]])
// CHECK: return %[[S2]]
// LOWER-LABEL: func.func @run_two_stage
// LOWER: %[[L1:.+]] = stablehlo.custom_call @__gpu$xla.gpu.triton(%arg0, %arg1)
// LOWER: %[[L2:.+]] = stablehlo.custom_call @__gpu$xla.gpu.triton(%[[L1]])
// LOWER: return %[[L2]]

// -----

// 7. enzymexla.gpu_wrapper raising to enzymexla.xla_wrapper + Triton lowering
module {
  func.func @gpu_wrap_triton(%a: memref<64xf32, 1>, %b: memref<64xf32, 1>) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 3.0 : f32
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c64, %c1, %c1) ({
      affine.parallel (%i) = (0) to (64) {
        %v = affine.load %a[%i] : memref<64xf32, 1>
        %r = arith.mulf %v, %cst : f32
        affine.store %r, %b[%i] : memref<64xf32, 1>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: enzymexla_tt_ext.module @rxla_gpu_0_tt
// CHECK: tt.func @rxla_gpu_0_kernel
// CHECK-LABEL: func.func @gpu_wrap_triton
// CHECK: enzymexla.xla_wrapper @rxla$raised_0 (%arg0, %arg1)
// CHECK-LABEL: func.func private @rxla$raised_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
// CHECK: enzymexla_tt_ext.call @rxla_gpu_0_tt::@rxla_gpu_0_inner::@rxla_gpu_0_kernel
// LOWER-LABEL: func.func @gpu_wrap_triton
// LOWER: enzymexla.xla_wrapper @rxla$raised_0 (%arg0, %arg1)
// LOWER-LABEL: func.func private @rxla$raised_0
// LOWER: stablehlo.custom_call @__gpu$xla.gpu.triton
