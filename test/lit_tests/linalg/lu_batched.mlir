// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
    %0:4 = enzymexla.linalg.lu %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
  }
}

// CPU:  llvm.func @enzymexla_lapack_sgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  llvm.func @enzymexla_lapack_sgetrf_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %2 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %3 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    llvm.store %0, %2 : i64, !llvm.ptr
// CPU-NEXT:    llvm.store %0, %3 : i64, !llvm.ptr
// CPU-NEXT:    llvm.call @enzymexla_lapack_sgetrf_(%2, %3, %arg0, %2, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// CPU:          %c_0 = stablehlo.constant dense<64> : tensor<i32>
// CPU-NEXT:     %c_1 = stablehlo.constant dense<1> : tensor<4x3x64xi64>
// CPU-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:     %c_3 = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:     %c_4 = stablehlo.constant dense<-1> : tensor<64xi64>
// CPU-NEXT:     %c_5 = stablehlo.constant dense<12> : tensor<i32>
// CPU-NEXT:     %c_6 = stablehlo.constant dense<-1> : tensor<12xi64>
// CPU-NEXT:     %c_7 = stablehlo.constant dense<-1> : tensor<12x64xi64>
// CPU-NEXT:     %c_8 = stablehlo.constant dense<0> : tensor<i32>
// CPU-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<4x3x64x64xf32>) -> tensor<12x64x64xf32>
// CPU-NEXT:     %1:4 = stablehlo.while(%iterArg = %c_8, %iterArg_9 = %0, %iterArg_10 = %c_7, %iterArg_11 = %c_6) : tensor<i32>, tensor<12x64x64xf32>, tensor<12x64xi64>, tensor<12xi64>
// CPU-NEXT:      cond {
// CPU-NEXT:       %11 = stablehlo.compare  LT, %iterArg, %c_5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CPU-NEXT:       stablehlo.return %11 : tensor<i1>
// CPU-NEXT:     } do {
// CPU-NEXT:       %11 = stablehlo.dynamic_slice %iterArg_9, %iterArg, %c_8, %c_8, sizes = [1, 64, 64] : (tensor<12x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x64x64xf32>
// CPU-NEXT:       %12 = stablehlo.reshape %11 : (tensor<1x64x64xf32>) -> tensor<64x64xf32>
// CPU-NEXT:       %13:3 = enzymexla.jit_call @enzymexla_lapack_sgetrf_wrapper_0 (%12, %c_4, %c_3) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>)
// CPU-NEXT:       %14 = stablehlo.reshape %13#0 : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
// CPU-NEXT:       %15 = stablehlo.dynamic_update_slice %iterArg_9, %14, %iterArg, %c_8, %c_8 : (tensor<12x64x64xf32>, tensor<1x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x64x64xf32>
// CPU-NEXT:       %16 = stablehlo.reshape %13#1 : (tensor<64xi64>) -> tensor<1x64xi64>
// CPU-NEXT:       %17 = stablehlo.dynamic_update_slice %iterArg_10, %16, %iterArg, %c_8 : (tensor<12x64xi64>, tensor<1x64xi64>, tensor<i32>, tensor<i32>) -> tensor<12x64xi64>
// CPU-NEXT:       %18 = stablehlo.reshape %13#2 : (tensor<i64>) -> tensor<1xi64>
// CPU-NEXT:       %19 = stablehlo.dynamic_update_slice %iterArg_11, %18, %iterArg : (tensor<12xi64>, tensor<1xi64>, tensor<i32>) -> tensor<12xi64>
// CPU-NEXT:       %20 = stablehlo.add %iterArg, %c_2 : tensor<i32>
// CPU-NEXT:       stablehlo.return %20, %15, %17, %19 : tensor<i32>, tensor<12x64x64xf32>, tensor<12x64xi64>, tensor<12xi64>
// CPU-NEXT:     }
// CPU-NEXT:     %2 = stablehlo.reshape %1#1 : (tensor<12x64x64xf32>) -> tensor<4x3x64x64xf32>
// CPU-NEXT:     %3 = stablehlo.reshape %1#2 : (tensor<12x64xi64>) -> tensor<4x3x64xi64>
// CPU-NEXT:     %4 = stablehlo.convert %1#3 : (tensor<12xi64>) -> tensor<12xi32>
// CPU-NEXT:     %5 = stablehlo.reshape %4 : (tensor<12xi32>) -> tensor<4x3xi32>
// CPU-NEXT:     %6 = stablehlo.subtract %3, %c_1 : tensor<4x3x64xi64>
// CPU-NEXT:     %7:2 = stablehlo.while(%iterArg = %c_8, %iterArg_9 = %c) : tensor<i32>, tensor<4x3x64xi64>
// CPU-NEXT:      cond {
// CPU-NEXT:       %11 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CPU-NEXT:       stablehlo.return %11 : tensor<i1>
// CPU-NEXT:     } do {
// CPU-NEXT:       %11 = stablehlo.add %iterArg, %c_2 : tensor<i32>
// CPU-NEXT:       %12 = stablehlo.dynamic_slice %6, %c_8, %c_8, %iterArg, sizes = [4, 3, 1] : (tensor<4x3x64xi64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x3x1xi64>
// CPU-NEXT:       %13 = stablehlo.dynamic_slice %iterArg_9, %c_8, %c_8, %iterArg, sizes = [4, 3, 1] : (tensor<4x3x64xi64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x3x1xi64>
// CPU-NEXT:       %14 = "stablehlo.gather"(%iterArg_9, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<4x3x64xi64>, tensor<4x3x1xi64>) -> tensor<4x3x1xi64>
// CPU-NEXT:       %15 = stablehlo.dynamic_update_slice %iterArg_9, %14, %c_8, %c_8, %iterArg : (tensor<4x3x64xi64>, tensor<4x3x1xi64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x3x64xi64>
// CPU-NEXT:       %16 = stablehlo.reshape %13 : (tensor<4x3x1xi64>) -> tensor<4x3xi64>
// CPU-NEXT:       %17 = "stablehlo.scatter"(%15, %12, %16) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 2>, unique_indices = false}> ({
// CPU-NEXT:       ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
// CPU-NEXT:         stablehlo.return %arg2 : tensor<i64>
// CPU-NEXT:       }) : (tensor<4x3x64xi64>, tensor<4x3x1xi64>, tensor<4x3xi64>) -> tensor<4x3x64xi64>
// CPU-NEXT:       stablehlo.return %11, %17 : tensor<i32>, tensor<4x3x64xi64>
// CPU-NEXT:     }
// CPU-NEXT:     %8 = stablehlo.add %7#1, %c_1 : tensor<4x3x64xi64>
// CPU-NEXT:     %9 = stablehlo.convert %3 : (tensor<4x3x64xi64>) -> tensor<4x3x64xi32>
// CPU-NEXT:     %10 = stablehlo.convert %8 : (tensor<4x3x64xi64>) -> tensor<4x3x64xi32>
// CPU-NEXT:     return %2, %9, %10, %5 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// CPU-NEXT:   }


// CUDA: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// CUDA-NEXT:    %c = stablehlo.constant dense<1> : tensor<4x3x64xi32>
// CUDA-NEXT:    %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {operand_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>)
// CUDA-NEXT:    %1 = stablehlo.subtract %0#1, %c : tensor<4x3x64xi32>
// CUDA-NEXT:    %2 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%1) : (tensor<4x3x64xi32>) -> tensor<4x3x64xi32>
// CUDA-NEXT:    %3 = stablehlo.add %c, %2 : tensor<4x3x64xi32>
// CUDA-NEXT:    return %0#0, %0#1, %3, %0#2 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// CUDA-NEXT:  }

// TPU: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// TPU-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<4x3x64xi32>
// TPU-NEXT:    %0:3 = stablehlo.custom_call @LUFactorization(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>)
// TPU-NEXT:    %1 = stablehlo.add %c_0, %0#1 : tensor<4x3x64xi32>
// TPU-NEXT:    %2 = stablehlo.add %c_0, %0#2 : tensor<4x3x64xi32>
// TPU-NEXT:    %3 = stablehlo.is_finite %0#0 : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xi1>
// TPU-NEXT:    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.and across dimensions = [2, 3] : (tensor<4x3x64x64xi1>, tensor<i1>) -> tensor<4x3xi1>
// TPU-NEXT:    %5 = stablehlo.not %4 : tensor<4x3xi1>
// TPU-NEXT:    %6 = stablehlo.convert %5 : (tensor<4x3xi1>) -> tensor<4x3xi32>
// TPU-NEXT:    return %0#0, %1, %2, %6 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// TPU-NEXT:  }
