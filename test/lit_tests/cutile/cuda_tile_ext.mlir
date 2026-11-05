// RUN: enzymexlamlir-opt %s --split-input-file -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: enzymexla_ct_ext.module @add_kernel_ct
  // CHECK: cuda_tile.module @add_kernel_inner
  // CHECK: entry @add_kernel(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>)
  enzymexla_ct_ext.module @add_kernel_ct {
    cuda_tile.module @add_kernel_inner {
      entry @add_kernel(%arg0: !cuda_tile.tile<ptr<f32>>, %arg1: !cuda_tile.tile<ptr<f32>>, %arg2: !cuda_tile.tile<ptr<f32>>) {
        %tok = make_token : token
        %bid_x, %bid_y, %bid_z = get_tile_block_id : tile<i32>
        %c128 = constant <i32: 128> : tile<i32>
        %base = muli %bid_x, %c128 : tile<i32>
        %base_1d = reshape %base : tile<i32> -> tile<1xi32>
        %base_bcast = broadcast %base_1d : tile<1xi32> -> tile<128xi32>
        %idx = iota : tile<128xi32>
        %off = addi %base_bcast, %idx : tile<128xi32>
        %a_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
        %a_ptrs = broadcast %a_1d : tile<1xptr<f32>> -> tile<128xptr<f32>>
        %a_off = offset %a_ptrs, %off : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
        %va, %tok_a = load_ptr_tko weak %a_off token = %tok : tile<128xptr<f32>> -> tile<128xf32>, token
        %b_1d = reshape %arg1 : tile<ptr<f32>> -> tile<1xptr<f32>>
        %b_ptrs = broadcast %b_1d : tile<1xptr<f32>> -> tile<128xptr<f32>>
        %b_off = offset %b_ptrs, %off : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
        %vb, %tok_b = load_ptr_tko weak %b_off token = %tok_a : tile<128xptr<f32>> -> tile<128xf32>, token
        %sum = addf %va, %vb : tile<128xf32>
        %c_1d = reshape %arg2 : tile<ptr<f32>> -> tile<1xptr<f32>>
        %c_ptrs = broadcast %c_1d : tile<1xptr<f32>> -> tile<128xptr<f32>>
        %c_off = offset %c_ptrs, %off : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
        %tok_c = store_ptr_tko weak %c_off, %sum token = %tok_b : tile<128xptr<f32>>, tile<128xf32> -> token
        return
      }
    }
  }

  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    // CHECK: enzymexla_ct_ext.call @add_kernel_ct::@add_kernel_inner::@add_kernel clusters in(%{{.*}}, %{{.*}}, %{{.*}}) blocks in(%{{.*}}, %{{.*}}, %{{.*}}) (%arg0, %arg1, %arg2)
    %0 = enzymexla_ct_ext.call @add_kernel_ct::@add_kernel_inner::@add_kernel clusters in (%c1, %c1, %c1) blocks in (%c8, %c1, %c1) (%arg0, %arg1, %arg2) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}

// -----

module {
  // CHECK-LABEL: enzymexla_ct_ext.module @llvm_kernel_ct
  // CHECK: llvm.func @llvm_kernel
  enzymexla_ct_ext.module @llvm_kernel_ct {
    llvm.func @llvm_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
      llvm.return
    }
  }

  // CHECK-LABEL: func.func @call_llvm
  func.func @call_llvm(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    // CHECK: enzymexla_ct_ext.call @llvm_kernel_ct::@llvm_kernel
    %0 = enzymexla_ct_ext.call @llvm_kernel_ct::@llvm_kernel clusters in (%c1, %c1, %c1) blocks in (%c1, %c1, %c1) (%arg0, %arg1) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
