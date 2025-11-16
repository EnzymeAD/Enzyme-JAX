// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
    %0:4 = enzymexla.linalg.lu %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
  }
}

// CPU: llvm.func private @enzymexla_lapack_sgetrf_wrapper(%arg0: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg1: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg2: !llvm.ptr {llvm.nofree}, %arg3: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg4: !llvm.ptr {llvm.nofree, llvm.writeonly}, %arg5: !llvm.ptr {llvm.nofree, llvm.writeonly}) {
// CPU-NEXT:    llvm.call @enzymexla_lapack_sgetrf_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU:  llvm.func @enzymexla_lapack_sgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// CPU:          %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:     %c_1 = stablehlo.constant dense<64> : tensor<i32>
// CPU-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<4x3x64xi64>
// CPU-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<i32>
// CPU-NEXT:     %0:3 = call @batched_enzymexla_lapack_sgetrf_[[WRAPPER_ID:[0-9]+]](%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi64>, tensor<4x3xi64>)
// CPU-NEXT:     %1 = stablehlo.subtract %0#1, %c_2 : tensor<4x3x64xi64>
// CPU-NEXT:     %2:2 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c) : tensor<i32>, tensor<4x3x64xi64>
// CPU-NEXT:     cond {
// CPU-NEXT:       %7 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CPU-NEXT:       stablehlo.return %7 : tensor<i1>
// CPU-NEXT:     } do {
// CPU-NEXT:       %7 = stablehlo.add %iterArg, %c_0 : tensor<i32>
// CPU-NEXT:       %8 = stablehlo.dynamic_slice %1, %c_3, %c_3, %iterArg, sizes = [4, 3, 1] : (tensor<4x3x64xi64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x3x1xi64>
// CPU-NEXT:       %9 = stablehlo.dynamic_slice %iterArg_4, %c_3, %c_3, %iterArg, sizes = [4, 3, 1] : (tensor<4x3x64xi64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x3x1xi64>
// CPU-NEXT:       %10 = "stablehlo.gather"(%iterArg_4, %8) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<4x3x64xi64>, tensor<4x3x1xi64>) -> tensor<4x3x1xi64>
// CPU-NEXT:       %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %c_3, %c_3, %iterArg : (tensor<4x3x64xi64>, tensor<4x3x1xi64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x3x64xi64>
// CPU-NEXT:       %12 = stablehlo.reshape %9 : (tensor<4x3x1xi64>) -> tensor<4x3xi64>
// CPU-NEXT:       %13 = "stablehlo.scatter"(%11, %8, %12) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 2>, unique_indices = false}> ({
// CPU-NEXT:       ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
// CPU-NEXT:         stablehlo.return %arg2 : tensor<i64>
// CPU-NEXT:       }) : (tensor<4x3x64xi64>, tensor<4x3x1xi64>, tensor<4x3xi64>) -> tensor<4x3x64xi64>
// CPU-NEXT:       stablehlo.return %7, %13 : tensor<i32>, tensor<4x3x64xi64>
// CPU-NEXT:     }
// CPU-NEXT:     %3 = stablehlo.add %2#1, %c_2 : tensor<4x3x64xi64>
// CPU-NEXT:     %4 = stablehlo.convert %0#1 : (tensor<4x3x64xi64>) -> tensor<4x3x64xi32>
// CPU-NEXT:     %5 = stablehlo.convert %3 : (tensor<4x3x64xi64>) -> tensor<4x3x64xi32>
// CPU-NEXT:     %6 = stablehlo.convert %0#2 : (tensor<4x3xi64>) -> tensor<4x3xi32>
// CPU-NEXT:     return %0#0, %4, %5, %6 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// CPU-NEXT:   }
// CPU-NEXT:   func.func private @batched_enzymexla_lapack_sgetrf_[[WRAPPER_ID]](%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi64>, tensor<4x3xi64>) {
// CPU-NEXT:     %c = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:     %c_0 = stablehlo.constant dense<-1> : tensor<64xi64>
// CPU-NEXT:     %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:     %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CPU-NEXT:     %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:     %c_4 = stablehlo.constant dense<12> : tensor<i64>
// CPU-NEXT:     %cst = arith.constant dense<0> : tensor<4x3xi64>
// CPU-NEXT:     %cst_5 = arith.constant dense<0> : tensor<4x3x64xi64>
// CPU-NEXT:     %cst_6 = arith.constant dense<0.000000e+00> : tensor<4x3x64x64xf32>
// CPU-NEXT:     %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:     %0:4 = stablehlo.while(%iterArg = %c_7, %iterArg_8 = %cst_6, %iterArg_9 = %cst_5, %iterArg_10 = %cst) : tensor<i64>, tensor<4x3x64x64xf32>, tensor<4x3x64xi64>, tensor<4x3xi64>
// CPU-NEXT:     cond {
// CPU-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT:       stablehlo.return %1 : tensor<i1>
// CPU-NEXT:     } do {
// CPU-NEXT:       %1 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CPU-NEXT:       %2 = stablehlo.remainder %iterArg, %c_2 : tensor<i64>
// CPU-NEXT:       %3 = stablehlo.divide %iterArg, %c_2 : tensor<i64>
// CPU-NEXT:       %4 = stablehlo.dynamic_slice %arg0, %2, %3, %c_7, %c_7, sizes = [1, 1, 64, 64] : (tensor<4x3x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x64xf32>
// CPU-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
// CPU-NEXT:       %6:3 = enzymexla.jit_call @enzymexla_lapack_sgetrf_wrapper (%c_1, %c_1, %5, %c_1, %c_0, %c) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<i64>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<64xi64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>)
// CPU-NEXT:       %7 = stablehlo.reshape %6#0 : (tensor<64x64xf32>) -> tensor<1x1x64x64xf32>
// CPU-NEXT:       %8 = stablehlo.dynamic_update_slice %iterArg_8, %7, %2, %3, %c_7, %c_7 : (tensor<4x3x64x64xf32>, tensor<1x1x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64x64xf32>
// CPU-NEXT:       %9 = stablehlo.reshape %6#1 : (tensor<64xi64>) -> tensor<1x1x64xi64>
// CPU-NEXT:       %10 = stablehlo.dynamic_update_slice %iterArg_9, %9, %2, %3, %c_7 : (tensor<4x3x64xi64>, tensor<1x1x64xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64xi64>
// CPU-NEXT:       %11 = stablehlo.reshape %6#2 : (tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT:       %12 = stablehlo.dynamic_update_slice %iterArg_10, %11, %2, %3 : (tensor<4x3xi64>, tensor<1x1xi64>, tensor<i64>, tensor<i64>) -> tensor<4x3xi64>
// CPU-NEXT:       stablehlo.return %1, %8, %10, %12 : tensor<i64>, tensor<4x3x64x64xf32>, tensor<4x3x64xi64>, tensor<4x3xi64>
// CPU-NEXT:     }
// CPU-NEXT:     return %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xi64>, tensor<4x3xi64>
// CPU-NEXT:   }


// CUDA: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// CUDA-NEXT:    %c = stablehlo.constant dense<1> : tensor<4x3x64xi32>
// CUDA-NEXT:    %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {api_version = 4 : i32, operand_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>)
// CUDA-NEXT:    %1 = stablehlo.subtract %0#1, %c : tensor<4x3x64xi32>
// CUDA-NEXT:    %2 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%1) {api_version = 4 : i32, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<4x3x64xi32>) -> tensor<4x3x64xi32>
// CUDA-NEXT:    %3 = stablehlo.add %c, %2 : tensor<4x3x64xi32>
// CUDA-NEXT:    return %0#0, %0#1, %3, %0#2 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// CUDA-NEXT:  }

// TPU: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// TPU-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<4x3x64xi32>
// TPU-NEXT:    %0:3 = stablehlo.custom_call @LuDecomposition(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>)
// TPU-NEXT:    %1 = stablehlo.add %c_0, %0#1 : tensor<4x3x64xi32>
// TPU-NEXT:    %2 = stablehlo.add %c_0, %0#2 : tensor<4x3x64xi32>
// TPU-NEXT:    %3 = stablehlo.is_finite %0#0 : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xi1>
// TPU-NEXT:    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.and across dimensions = [2, 3] : (tensor<4x3x64x64xi1>, tensor<i1>) -> tensor<4x3xi1>
// TPU-NEXT:    %5 = stablehlo.not %4 : tensor<4x3xi1>
// TPU-NEXT:    %6 = stablehlo.convert %5 : (tensor<4x3xi1>) -> tensor<4x3xi32>
// TPU-NEXT:    return %0#0, %1, %2, %6 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// TPU-NEXT:  }
