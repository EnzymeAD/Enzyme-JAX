// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-mosaic | FileCheck %s

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

// CHECK-LABEL: enzymexla_mosaic_ext.module @vec_add_mosaic
// CHECK: builtin.module @vec_add_inner
// CHECK: func.func @vec_add_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<256xf32, #tpu.memory_space<vmem>>, %arg4: memref<256xf32, #tpu.memory_space<vmem>>, %arg5: memref<256xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>], iteration_bounds = array<i64: 2, 1, 1>, tpu.core_type = #tpu.core_type<tc>}
// CHECK: tpu.iota {dimensions = array<i32: 0>} : vector<128xi64>
// CHECK: tpu.vector_load %arg3[%{{.*}}] {{.*}}: memref<256xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
// CHECK: tpu.vector_load %arg4[%{{.*}}] {{.*}}: memref<256xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
// CHECK: arith.addf {{.*}} : vector<128xf32>
// CHECK: tpu.vector_store %arg5[%{{.*}}], %{{.*}} : memref<256xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
// CHECK: return
// CHECK-LABEL: func.func @main
// CHECK: enzymexla_mosaic_ext.call @vec_add_mosaic::@vec_add_inner::@vec_add_kernel

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

// CHECK-LABEL: enzymexla_mosaic_ext.module @mat_row_scale_mosaic
// CHECK: builtin.module @mat_row_scale_inner
// CHECK: func.func @mat_row_scale_kernel
// CHECK: memref.collapse_shape {{.*}} {{\[\[}}0, 1{{\]\]}} : memref<20x36xf32, #tpu.memory_space<vmem>> into memref<720xf32, #tpu.memory_space<vmem>>
// CHECK: memref.collapse_shape {{.*}} {{\[\[}}0, 1{{\]\]}} : memref<36x12xf32, #tpu.memory_space<vmem>> into memref<432xf32, #tpu.memory_space<vmem>>
// CHECK: memref.collapse_shape {{.*}} {{\[\[}}0, 1{{\]\]}} : memref<20x12xf32, #tpu.memory_space<vmem>> into memref<240xf32, #tpu.memory_space<vmem>>
// CHECK: scf.for {{.*}} -> (vector<32x16xf32>)
// CHECK: tpu.vector_load_idx {{.*}} masked {{.*}} : memref<720xf32, #tpu.memory_space<vmem>>[vector<32x16xi32>], vector<32x16xf32>, vector<32x16xi1>
// CHECK: arith.mulf {{.*}} : vector<32x16xf32>
// CHECK: arith.addf {{.*}} : vector<32x16xf32>
// CHECK: tpu.vector_store_idx {{.*}} masked {{.*}} : memref<240xf32, #tpu.memory_space<vmem>>[vector<32x16xi32>], vector<32x16xf32>, vector<32x16xi1>
// CHECK-LABEL: func.func @entry_2d
// CHECK: enzymexla_mosaic_ext.call @mat_row_scale_mosaic::@mat_row_scale_inner::@mat_row_scale_kernel

// -----

// 3. Standalone memref function + enzyme.affine_atomic_rmw scatter-add
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

// CHECK-LABEL: enzymexla_mosaic_ext.module @atomic_scatter_mosaic
// CHECK: func.func @atomic_scatter_kernel
// CHECK: tpu.vector_store_idx {{.*}} {add = true} : memref<8xf32, #tpu.memory_space<vmem>>[vector<32xi32>], vector<32xf32>,
// CHECK-LABEL: func.func private @atomic_scatter_raised(%arg0: tensor<8xf32>, %arg1: tensor<32xi32>, %arg2: tensor<32xf32>) -> (tensor<8xf32>, tensor<32xi32>, tensor<32xf32>)
// CHECK: %[[UPDATED:.+]] = enzymexla_mosaic_ext.call @atomic_scatter_mosaic::@atomic_scatter_inner::@atomic_scatter_kernel
// CHECK: return %[[UPDATED]], %arg1, %arg2

// -----

// 4. 1D affine.for reduction vectorized via tpu.reduce
module {
  func.func private @vec_reduce_sum(%x: memref<128xf32, 1>, %out: memref<1xf32, 1>) {
    %zero = arith.constant 0.0 : f32
    %sum = affine.for %i = 0 to 128 iter_args(%acc = %zero) -> (f32) {
      %v = affine.load %x[%i] : memref<128xf32, 1>
      %next = arith.addf %acc, %v : f32
      affine.yield %next : f32
    }
    %c0 = arith.constant 0 : index
    affine.store %sum, %out[%c0] : memref<1xf32, 1>
    return
  }

  func.func @run_reduce(%x: tensor<128xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
    %r = enzymexla.jit_call @vec_reduce_sum(%x, %out) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<128xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %r : tensor<1xf32>
  }
}

// CHECK-LABEL: enzymexla_mosaic_ext.module @vec_reduce_sum_mosaic
// CHECK: func.func @vec_reduce_sum_kernel
// CHECK: tpu.vector_load {{.*}} : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
// CHECK: tpu.reduce <sum>, {{.*}} [0] : vector<128xf32> -> vector<1xf32>
// CHECK: vector.extract {{.*}}[0] : f32 from vector<1xf32>
// CHECK: memref.store {{.*}} : memref<1xf32, #tpu.memory_space<vmem>>
// CHECK-LABEL: func.func @run_reduce
// CHECK: enzymexla_mosaic_ext.call @vec_reduce_sum_mosaic::@vec_reduce_sum_inner::@vec_reduce_sum_kernel

// -----

// 5. Dynamic parallel bound read from rank-0 scalar memref (grid-stride loop)
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

// CHECK-LABEL: enzymexla_mosaic_ext.module @dyn_scale_mosaic
// CHECK: func.func @dyn_scale_kernel
// CHECK: scf.for
// CHECK: tpu.vector_load {{.*}} masked {{.*}} : memref<512xf32, #tpu.memory_space<vmem>>, vector<128xf32>, vector<128xi1>
// CHECK: tpu.vector_store {{.*}} masked {{.*}} : memref<512xf32, #tpu.memory_space<vmem>>, vector<128xf32>, vector<128xi1>
// CHECK-LABEL: func.func @run_dyn
// CHECK: enzymexla_mosaic_ext.call @dyn_scale_mosaic::@dyn_scale_inner::@dyn_scale_kernel

// -----

// 6. enzymexla.gpu_wrapper raising to enzymexla.xla_wrapper
module {
  func.func @gpu_wrap_test(%a: memref<64xf32, 1>, %b: memref<64xf32, 1>) {
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

// CHECK-LABEL: enzymexla_mosaic_ext.module @rxla_gpu_0_mosaic
// CHECK: func.func @rxla_gpu_0_kernel
// CHECK-LABEL: func.func @gpu_wrap_test
// CHECK: enzymexla.xla_wrapper @rxla$raised_0 (%arg0, %arg1)
// CHECK-LABEL: func.func private @rxla$raised_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
// CHECK: enzymexla_mosaic_ext.call @rxla_gpu_0_mosaic::@rxla_gpu_0_inner::@rxla_gpu_0_kernel

// -----

// 7. Sequential multi-launch segmentation + enzymexla.jit_region
module {
  func.func @region_two_stage(%in: tensor<4x16xf32>, %out: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %res = "enzymexla.jit_region"(%in, %out) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> ({
    ^bb0(%m_in: memref<4x16xf32>, %m_out: memref<4x16xf32>):
      %one = arith.constant 1.0 : f32
      %two = arith.constant 2.0 : f32
      affine.parallel (%i, %j) = (0, 0) to (4, 16) {
        %va = affine.load %m_in[%i, %j] : memref<4x16xf32>
        %s1 = arith.addf %va, %one : f32
        affine.store %s1, %m_out[%i, %j] : memref<4x16xf32>
      }
      affine.parallel (%i, %j) = (0, 0) to (4, 16) {
        %vb = affine.load %m_out[%i, %j] : memref<4x16xf32>
        %s2 = arith.mulf %vb, %two : f32
        affine.store %s2, %m_out[%i, %j] : memref<4x16xf32>
      }
      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}

// CHECK-DAG: enzymexla_mosaic_ext.module @jit_region_0_0_mosaic
// CHECK-DAG: enzymexla_mosaic_ext.module @jit_region_0_1_mosaic
// CHECK-LABEL: func.func @region_two_stage
// CHECK: %[[C1:.+]] = enzymexla_mosaic_ext.call @jit_region_0_0_mosaic::@jit_region_0_0_inner::@jit_region_0_0_kernel {{.*}}(%arg0, %arg1)
// CHECK: %[[C2:.+]] = enzymexla_mosaic_ext.call @jit_region_0_1_mosaic::@jit_region_0_1_inner::@jit_region_0_1_kernel {{.*}}(%[[C1]])
// CHECK: return %[[C2]]
