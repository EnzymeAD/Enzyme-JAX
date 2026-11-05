// RUN: enzymexlamlir-opt %s --split-input-file -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: enzymexla_mosaic_ext.module @add_kernel_mosaic
  // CHECK: builtin.module @add_kernel_inner
  // CHECK: func.func public @add_kernel(%arg0: i32, %arg1: memref<1024xf32, #tpu.memory_space<vmem>>, %arg2: memref<1024xf32, #tpu.memory_space<vmem>>, %arg3: memref<1024xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<parallel>], iteration_bounds = array<i64: 8>, tpu.core_type = #tpu.core_type<tc>}
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
        // CHECK: tpu.iota {dimensions = array<i32: 0>} : vector<128xi32>
        %iota = tpu.iota {dimensions = array<i32: 0>} : vector<128xi32>
        %bvec = vector.broadcast %base_i32 : i32 to vector<128xi32>
        %idx = arith.addi %bvec, %iota : vector<128xi32>
        %base = arith.index_cast %base_i32 : i32 to index
        // CHECK: tpu.vector_load
        %va = tpu.vector_load %arg1[%base] {strides = array<i32>} : memref<1024xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
        // CHECK: tpu.vector_load_idx
        %vb = tpu.vector_load_idx %arg2[%idx] : memref<1024xf32, #tpu.memory_space<vmem>>[vector<128xi32>], vector<128xf32>,
        %sum = arith.addf %va, %vb : vector<128xf32>
        // CHECK: tpu.vector_store
        tpu.vector_store %arg3[%base], %sum {strides = array<i32>} : memref<1024xf32, #tpu.memory_space<vmem>>, vector<128xf32>,
        return
      }
    }
  }

  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    // CHECK: enzymexla_mosaic_ext.call @add_kernel_mosaic::@add_kernel_inner::@add_kernel clusters in(%{{.*}}, %{{.*}}, %{{.*}}) blocks in(%{{.*}}, %{{.*}}, %{{.*}}) (%arg0, %arg1, %arg2)
    %0 = enzymexla_mosaic_ext.call @add_kernel_mosaic::@add_kernel_inner::@add_kernel clusters in(%c1, %c1, %c1) blocks in(%c8, %c1, %c1) (%arg0, %arg1, %arg2) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}

// -----

module {
  // CHECK-LABEL: enzymexla_mosaic_ext.module @llvm_kernel_mosaic
  // CHECK: llvm.func @llvm_kernel
  enzymexla_mosaic_ext.module @llvm_kernel_mosaic {
    llvm.func @llvm_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
      llvm.return
    }
  }

  // CHECK-LABEL: func.func @call_llvm
  func.func @call_llvm(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    // CHECK: enzymexla_mosaic_ext.call @llvm_kernel_mosaic::@llvm_kernel
    %0 = enzymexla_mosaic_ext.call @llvm_kernel_mosaic::@llvm_kernel clusters in(%c1, %c1, %c1) blocks in(%c1, %c1, %c1) (%arg0, %arg1) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
