// RUN: enzymexlamlir-opt %s --split-input-file -lower-mosaic | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --split-input-file -lower-mosaic | enzymexlamlir-opt --split-input-file --raise-mosaic-custom-call -canonicalize | FileCheck %s --check-prefix=ROUNDTRIP

module {
  // LOWER-NOT: enzymexla_mosaic_ext.module
  // ROUNDTRIP-LABEL: func.func @main
  enzymexla_mosaic_ext.module @add_kernel_mosaic {
    builtin.module @add_kernel_inner {
      func.func public @add_kernel(
          %arg0: i32,
          %arg1: memref<1024xf32, #tpu.memory_space<vmem>>,
          %arg2: memref<1024xf32, #tpu.memory_space<vmem>>,
          %arg3: memref<1024xf32, #tpu.memory_space<vmem>>)
          attributes {
            dimension_semantics = [#tpu.dimension_semantics<parallel>],
            iteration_bounds = array<i64: 8>,
            tpu.core_type = #tpu.core_type<tc>
          } {
        %c128_i32 = arith.constant 128 : i32
        %base_i32 = arith.muli %arg0, %c128_i32 : i32
        %iota = tpu.iota {dimensions = array<i32: 0>} : vector<128xi32>
        %bvec = vector.broadcast %base_i32 : i32 to vector<128xi32>
        %idx = arith.addi %bvec, %iota : vector<128xi32>
        %base = arith.index_cast %base_i32 : i32 to index
        %va = tpu.vector_load %arg1[%base] {strides = array<i32>} : memref<1024xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
        %vb = tpu.vector_load_idx %arg2[%idx] : memref<1024xf32, #tpu.memory_space<vmem>>[vector<128xi32>], vector<128xf32>,
        %sum = arith.addf %va, %vb : vector<128xf32>
        tpu.vector_store %arg3[%base], %sum {strides = array<i32>} : memref<1024xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
        return
      }
    }
  }

  // LOWER-LABEL: func.func @main
  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    // LOWER: %[[RES:.*]] = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1, %arg2)
    // LOWER-SAME: backend_config = "{\22custom_call_config\22:{\22body\22:\22{{[A-Za-z0-9+/=]+}}\22,\22needs_layout_passes\22:true,\22serialization_format\22:1}}"
    // LOWER-SAME: has_side_effect = true
    // LOWER-SAME: kernel_name = "add_kernel"
    // LOWER-SAME: operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]
    // LOWER-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]
    // LOWER-SAME: result_layouts = [dense<0> : tensor<1xindex>]
    // LOWER-NEXT: return %[[RES]] : tensor<1024xf32>
    // ROUNDTRIP: %[[CALL:.*]] = enzymexla_mosaic_ext.call @mosaic_module::@mosaic_module_inner::@add_kernel clusters in(%{{.*}}, %{{.*}}, %{{.*}}) blocks in(%{{.*}}, %{{.*}}, %{{.*}}) (%arg0, %arg1, %arg2)
    // ROUNDTRIP-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]
    // ROUNDTRIP: return %[[CALL]] : tensor<1024xf32>
    %0 = enzymexla_mosaic_ext.call @add_kernel_mosaic::@add_kernel_inner::@add_kernel clusters in(%c1, %c1, %c1) blocks in(%c8, %c1, %c1) (%arg0, %arg1, %arg2) <{operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  // ROUNDTRIP-LABEL: enzymexla_mosaic_ext.module @mosaic_module
  // ROUNDTRIP: builtin.module @mosaic_module_inner
  // ROUNDTRIP: func.func @add_kernel(%arg0: i32, %arg1: memref<1024xf32, #tpu.memory_space<vmem>>, %arg2: memref<1024xf32, #tpu.memory_space<vmem>>, %arg3: memref<1024xf32, #tpu.memory_space<vmem>>, %arg4: memref<1024xf32, #tpu.memory_space<vmem>>)
  // ROUNDTRIP-NOT: scf.if
  // ROUNDTRIP: tpu.iota
  // ROUNDTRIP: tpu.vector_load %arg1
  // ROUNDTRIP: tpu.vector_load_idx %arg2
  // ROUNDTRIP: tpu.vector_store %arg4
}

// -----

module {
  // In-place accumulating kernel: `%arg2` is read/accumulated (`add = true`),
  // so `-lower-mosaic` appends output `%arg3` and initializes `%arg3` from
  // `%arg2` on grid iteration 0 before accumulating into `%arg3`.
  // LOWER-NOT: enzymexla_mosaic_ext.module
  // ROUNDTRIP-LABEL: func.func @accum_caller
  enzymexla_mosaic_ext.module @accum_mosaic {
    builtin.module @accum_inner {
      func.func public @accum_kernel(
          %arg0: i32,
          %arg1: memref<256xf32, #tpu.memory_space<vmem>>,
          %arg2: memref<256xf32, #tpu.memory_space<vmem>>)
          attributes {
            dimension_semantics = [#tpu.dimension_semantics<parallel>],
            iteration_bounds = array<i64: 2>,
            tpu.core_type = #tpu.core_type<tc>
          } {
        %c128_i32 = arith.constant 128 : i32
        %off_i32 = arith.muli %arg0, %c128_i32 : i32
        %off = arith.index_cast %off_i32 : i32 to index
        %v = tpu.vector_load %arg1[%off] {strides = array<i32>} : memref<256xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
        tpu.vector_store %arg2[%off], %v {add = true, strides = array<i32>} : memref<256xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
        return
      }
    }
  }

  // LOWER-LABEL: func.func @accum_caller
  func.func @accum_caller(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    // LOWER: stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
    // LOWER-SAME: kernel_name = "accum_kernel"
    // LOWER-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]
    // ROUNDTRIP: enzymexla_mosaic_ext.call @mosaic_module::@mosaic_module_inner::@accum_kernel
    %0 = enzymexla_mosaic_ext.call @accum_mosaic::@accum_inner::@accum_kernel clusters in(%c1, %c1, %c1) blocks in(%c2, %c1, %c1) (%arg0, %arg1) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  // ROUNDTRIP-LABEL: enzymexla_mosaic_ext.module @mosaic_module
  // ROUNDTRIP: func.func @accum_kernel(%arg0: i32, %arg1: memref<256xf32, #tpu.memory_space<vmem>>, %arg2: memref<256xf32, #tpu.memory_space<vmem>>, %arg3: memref<256xf32, #tpu.memory_space<vmem>>)
  // ROUNDTRIP: scf.if
  // ROUNDTRIP: vector.load %arg2
  // ROUNDTRIP: vector.store %{{.*}}, %arg3
  // ROUNDTRIP: tpu.vector_store %arg3[%{{.*}}], %{{.*}} {add = true
}

// -----

module {
  // SparseCore + pure (`xla_side_effect_free`) + custom `backend_config` JSON.
  // LOWER-NOT: enzymexla_mosaic_ext.module
  // ROUNDTRIP-LABEL: func.func @sc_caller
  enzymexla_mosaic_ext.module @sc_mosaic {
    builtin.module @sc_inner {
      func.func public @sc_copy(
          %arg0: memref<64xf32, #tpu.memory_space<vmem>>,
          %arg1: memref<64xf32, #tpu.memory_space<vmem>>)
          attributes {
            dimension_semantics = [],
            tpu.core_type = #tpu.core_type<sc_vector_subcore>
          } {
        %c0 = arith.constant 0 : index
        %v = vector.load %arg0[%c0] : memref<64xf32, #tpu.memory_space<vmem>>, vector<64xf32>
        vector.store %v, %arg1[%c0] : memref<64xf32, #tpu.memory_space<vmem>>, vector<64xf32>
        return
      }
    }
  }

  // LOWER-LABEL: func.func @sc_caller
  func.func @sc_caller(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    // LOWER: stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
    // LOWER-SAME: internal_scratch_in_bytes
    // LOWER-SAME: DEVICE_TYPE_SPARSECORE
    // LOWER-SAME: sparse_core_config
    // LOWER-NOT: has_side_effect = true
    // LOWER-SAME: kernel_name = "sc_copy"
    // ROUNDTRIP: enzymexla_mosaic_ext.call @mosaic_module::@mosaic_module_inner::@sc_copy
    // ROUNDTRIP-SAME: backend_config = "{\22custom_call_config\22:{\22internal_scratch_in_bytes\22:4096}}"
    // ROUNDTRIP-SAME: xla_side_effect_free
    %0 = enzymexla_mosaic_ext.call @sc_mosaic::@sc_inner::@sc_copy clusters in(%c1, %c1, %c1) blocks in(%c1, %c1, %c1) (%arg0, %arg1) <{backend_config = "{\22custom_call_config\22:{\22internal_scratch_in_bytes\22:4096}}", output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], xla_side_effect_free}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  // ROUNDTRIP-LABEL: enzymexla_mosaic_ext.module @mosaic_module
  // ROUNDTRIP: func.func @sc_copy(%arg0: memref<64xf32, #tpu.memory_space<vmem>>, %arg1: memref<64xf32, #tpu.memory_space<vmem>>, %arg2: memref<64xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], tpu.core_type = #tpu.core_type<sc_vector_subcore>}
}
