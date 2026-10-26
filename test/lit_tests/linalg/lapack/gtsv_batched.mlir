// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=cpu blas_int_width=64" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=cuda" %s | FileCheck %s --check-prefix=CUDA

func.func @main(%dl: tensor<4x3x64xf64>, %d: tensor<4x3x64xf64>, %du: tensor<4x3x64xf64>, %b: tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi32>) {
  %0:2 = enzymexla.lapack.gtsv %dl, %d, %du, %b : (tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi32>)
  return %0#0, %0#1 : tensor<4x3x64x1xf64>, tensor<4x3xi32>
}

// CPU: func.func private @enzymexla_lapack_dgtsv_0(%[[ARG0:[^:]+]]: tensor<64xf64>, %[[ARG1:[^:]+]]: tensor<64xf64>, %[[ARG2:[^:]+]]: tensor<64xf64>, %[[ARG3:[^:]+]]: tensor<64x1xf64>) -> (tensor<64x1xf64>, tensor<i64>) {
// CPU-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_2:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT: %[[JIT_CALL_0:.*]]:5 = enzymexla.jit_call @enzymexla_lapack_dgtsv_wrapper (%[[CONSTANT_0]], %[[CONSTANT_1]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[CONSTANT_0]], %[[CONSTANT_2]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 7, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64x1xf64>, tensor<i64>, tensor<i64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64x1xf64>, tensor<i64>)
// CPU-NEXT: return %[[JIT_CALL_0]]#3, %[[JIT_CALL_0]]#4 : tensor<64x1xf64>, tensor<i64>
// CPU-NEXT: }

// CPU: llvm.func private @enzymexla_lapack_dgtsv_wrapper(%[[ARG0:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG2:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG3:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG4:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG5:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG6:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG7:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.writeonly}) {
// CPU-NEXT: %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[ARG2]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CPU-NEXT: llvm.call @enzymexla_lapack_dgtsv_(%[[ARG0]], %[[ARG1]], %[[GETELEMENTPTR_0]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT: llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_lapack_dgtsv_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU: func.func @main(%[[ARG0:[^:]+]]: tensor<4x3x64xf64>, %[[ARG1:[^:]+]]: tensor<4x3x64xf64>, %[[ARG2:[^:]+]]: tensor<4x3x64xf64>, %[[ARG3:[^:]+]]: tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi32>) {
// CPU-NEXT: %[[VAL_0:.*]]:2 = call @batched_enzymexla_lapack_dgtsv_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi64>)
// CPU-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[VAL_0]]#1 : (tensor<4x3xi64>) -> tensor<4x3xi32>
// CPU-NEXT: return %[[VAL_0]]#0, %[[CONVERT_0]] : tensor<4x3x64x1xf64>, tensor<4x3xi32>
// CPU-NEXT: }

// CPU: func.func private @batched_enzymexla_lapack_dgtsv_0(%[[ARG0:[^:]+]]: tensor<4x3x64xf64>, %[[ARG1:[^:]+]]: tensor<4x3x64xf64>, %[[ARG2:[^:]+]]: tensor<4x3x64xf64>, %[[ARG3:[^:]+]]: tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi64>) {
// CPU-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<3> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<4> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_3:.*]] = stablehlo.constant dense<12> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_4:.*]] = stablehlo.constant dense<64> : tensor<4x3xi64>
// CPU-NEXT: %[[CONSTANT_5:.*]] = stablehlo.constant dense<1> : tensor<4x3xi64>
// CPU-NEXT: %[[CONSTANT_6:.*]] = stablehlo.constant dense<-1> : tensor<4x3xi64>
// CPU-NEXT: %[[CONSTANT_7:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_8:.*]] = arith.constant dense<0.000000e+00> : tensor<4x3x64xf64>
// CPU-NEXT: %[[CONSTANT_9:.*]] = arith.constant dense<0.000000e+00> : tensor<4x3x64x1xf64>
// CPU-NEXT: %[[CONSTANT_10:.*]] = arith.constant dense<0> : tensor<4x3xi64>
// CPU-NEXT: %[[WHILE_0:.*]]:6 = stablehlo.while(%[[VAL_0:.*]] = %[[CONSTANT_7]], %[[VAL_1:.*]] = %[[CONSTANT_8]], %[[VAL_2:.*]] = %[[CONSTANT_8]], %[[VAL_3:.*]] = %[[CONSTANT_8]], %[[VAL_4:.*]] = %[[CONSTANT_9]], %[[VAL_5:.*]] = %[[CONSTANT_10]]) : tensor<i64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>, tensor<4x3xi64>
// CPU-NEXT: cond {
// CPU-NEXT: %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_0]], %[[CONSTANT_3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT: stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CPU-NEXT: } do {
// CPU-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[VAL_0]], %[[CONSTANT_2]] : tensor<i64>
// CPU-NEXT: %[[DIVIDE_0:.*]] = stablehlo.divide %[[VAL_0]], %[[CONSTANT_2]] : tensor<i64>
// CPU-NEXT: %[[REMAINDER_0:.*]] = stablehlo.remainder %[[DIVIDE_0]], %[[CONSTANT_1]] : tensor<i64>
// CPU-NEXT: %[[DIVIDE_1:.*]] = stablehlo.divide %[[VAL_0]], %[[CONSTANT_1]] : tensor<i64>
// CPU-NEXT: %[[REMAINDER_1:.*]] = stablehlo.remainder %[[DIVIDE_1]], %[[CONSTANT_0]] : tensor<i64>
// CPU-NEXT: %[[DYNAMIC_SLICE_0:.*]] = stablehlo.dynamic_slice %[[CONSTANT_4]], %[[REMAINDER_0]], %[[REMAINDER_1]], sizes = [1, 1] : (tensor<4x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_0]] : (tensor<1x1xi64>) -> tensor<i64>
// CPU-NEXT: %[[DYNAMIC_SLICE_1:.*]] = stablehlo.dynamic_slice %[[CONSTANT_5]], %[[REMAINDER_0]], %[[REMAINDER_1]], sizes = [1, 1] : (tensor<4x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_1]] : (tensor<1x1xi64>) -> tensor<i64>
// CPU-NEXT: %[[DYNAMIC_SLICE_2:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]], sizes = [1, 1, 64] : (tensor<4x3x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64xf64>
// CPU-NEXT: %[[RESHAPE_2:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_2]] : (tensor<1x1x64xf64>) -> tensor<64xf64>
// CPU-NEXT: %[[DYNAMIC_SLICE_3:.*]] = stablehlo.dynamic_slice %[[ARG1]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]], sizes = [1, 1, 64] : (tensor<4x3x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64xf64>
// CPU-NEXT: %[[RESHAPE_3:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_3]] : (tensor<1x1x64xf64>) -> tensor<64xf64>
// CPU-NEXT: %[[DYNAMIC_SLICE_4:.*]] = stablehlo.dynamic_slice %[[ARG2]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]], sizes = [1, 1, 64] : (tensor<4x3x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64xf64>
// CPU-NEXT: %[[RESHAPE_4:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_4]] : (tensor<1x1x64xf64>) -> tensor<64xf64>
// CPU-NEXT: %[[DYNAMIC_SLICE_5:.*]] = stablehlo.dynamic_slice %[[ARG3]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]], %[[CONSTANT_7]], sizes = [1, 1, 64, 1] : (tensor<4x3x64x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x1xf64>
// CPU-NEXT: %[[RESHAPE_5:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_5]] : (tensor<1x1x64x1xf64>) -> tensor<64x1xf64>
// CPU-NEXT: %[[DYNAMIC_SLICE_6:.*]] = stablehlo.dynamic_slice %[[CONSTANT_4]], %[[REMAINDER_0]], %[[REMAINDER_1]], sizes = [1, 1] : (tensor<4x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT: %[[RESHAPE_6:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_6]] : (tensor<1x1xi64>) -> tensor<i64>
// CPU-NEXT: %[[DYNAMIC_SLICE_7:.*]] = stablehlo.dynamic_slice %[[CONSTANT_6]], %[[REMAINDER_0]], %[[REMAINDER_1]], sizes = [1, 1] : (tensor<4x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT: %[[RESHAPE_7:.*]] = stablehlo.reshape %[[DYNAMIC_SLICE_7]] : (tensor<1x1xi64>) -> tensor<i64>
// CPU-NEXT: %[[JIT_CALL_0:.*]]:5 = enzymexla.jit_call @enzymexla_lapack_dgtsv_wrapper (%[[RESHAPE_0]], %[[RESHAPE_1]], %[[RESHAPE_2]], %[[RESHAPE_3]], %[[RESHAPE_4]], %[[RESHAPE_5]], %[[RESHAPE_6]], %[[RESHAPE_7]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 7, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64x1xf64>, tensor<i64>, tensor<i64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64x1xf64>, tensor<i64>)
// CPU-NEXT: %[[RESHAPE_8:.*]] = stablehlo.reshape %[[JIT_CALL_0]]#0 : (tensor<64xf64>) -> tensor<1x1x64xf64>
// CPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_0:.*]] = stablehlo.dynamic_update_slice %[[VAL_1]], %[[RESHAPE_8]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]] : (tensor<4x3x64xf64>, tensor<1x1x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64xf64>
// CPU-NEXT: %[[RESHAPE_9:.*]] = stablehlo.reshape %[[JIT_CALL_0]]#1 : (tensor<64xf64>) -> tensor<1x1x64xf64>
// CPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_1:.*]] = stablehlo.dynamic_update_slice %[[VAL_2]], %[[RESHAPE_9]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]] : (tensor<4x3x64xf64>, tensor<1x1x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64xf64>
// CPU-NEXT: %[[RESHAPE_10:.*]] = stablehlo.reshape %[[JIT_CALL_0]]#2 : (tensor<64xf64>) -> tensor<1x1x64xf64>
// CPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_2:.*]] = stablehlo.dynamic_update_slice %[[VAL_3]], %[[RESHAPE_10]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]] : (tensor<4x3x64xf64>, tensor<1x1x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64xf64>
// CPU-NEXT: %[[RESHAPE_11:.*]] = stablehlo.reshape %[[JIT_CALL_0]]#3 : (tensor<64x1xf64>) -> tensor<1x1x64x1xf64>
// CPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_3:.*]] = stablehlo.dynamic_update_slice %[[VAL_4]], %[[RESHAPE_11]], %[[REMAINDER_0]], %[[REMAINDER_1]], %[[CONSTANT_7]], %[[CONSTANT_7]] : (tensor<4x3x64x1xf64>, tensor<1x1x64x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64x1xf64>
// CPU-NEXT: %[[RESHAPE_12:.*]] = stablehlo.reshape %[[JIT_CALL_0]]#4 : (tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_4:.*]] = stablehlo.dynamic_update_slice %[[VAL_5]], %[[RESHAPE_12]], %[[REMAINDER_0]], %[[REMAINDER_1]] : (tensor<4x3xi64>, tensor<1x1xi64>, tensor<i64>, tensor<i64>) -> tensor<4x3xi64>
// CPU-NEXT: stablehlo.return %[[ADD_0]], %[[DYNAMIC_UPDATE_SLICE_0]], %[[DYNAMIC_UPDATE_SLICE_1]], %[[DYNAMIC_UPDATE_SLICE_2]], %[[DYNAMIC_UPDATE_SLICE_3]], %[[DYNAMIC_UPDATE_SLICE_4]] : tensor<i64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>, tensor<4x3xi64>
// CPU-NEXT: }
// CPU-NEXT: return %[[WHILE_0]]#4, %[[WHILE_0]]#5 : tensor<4x3x64x1xf64>, tensor<4x3xi64>
// CPU-NEXT: }

// CUDA: func.func @main(%[[ARG0:[^:]+]]: tensor<4x3x64xf64>, %[[ARG1:[^:]+]]: tensor<4x3x64xf64>, %[[ARG2:[^:]+]]: tensor<4x3x64xf64>, %[[ARG3:[^:]+]]: tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi32>) {
// CUDA-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CUDA-NEXT: %[[CUSTOM_CALL_0:.*]] = stablehlo.custom_call @cusparse_gtsv2_ffi(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) {api_version = 4 : i32, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>]} : (tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>) -> tensor<4x3x64x1xf64>
// CUDA-NEXT: %[[REAL_0:.*]] = stablehlo.real %[[CUSTOM_CALL_0]] : tensor<4x3x64x1xf64>
// CUDA-NEXT: %[[IS_FINITE_0:.*]] = stablehlo.is_finite %[[REAL_0]] : (tensor<4x3x64x1xf64>) -> tensor<4x3x64x1xi1>
// CUDA-NEXT: %[[IMAG_0:.*]] = stablehlo.imag %[[CUSTOM_CALL_0]] : tensor<4x3x64x1xf64>
// CUDA-NEXT: %[[IS_FINITE_1:.*]] = stablehlo.is_finite %[[IMAG_0]] : (tensor<4x3x64x1xf64>) -> tensor<4x3x64x1xi1>
// CUDA-NEXT: %[[AND_0:.*]] = stablehlo.and %[[IS_FINITE_0]], %[[IS_FINITE_1]] : tensor<4x3x64x1xi1>
// CUDA-NEXT: %[[REDUCE_0:.*]] = stablehlo.reduce(%[[AND_0]] init: %[[CONSTANT_0]]) applies stablehlo.and across dimensions = [2, 3] : (tensor<4x3x64x1xi1>, tensor<i1>) -> tensor<4x3xi1>
// CUDA-NEXT: %[[NOT_0:.*]] = stablehlo.not %[[REDUCE_0]] : tensor<4x3xi1>
// CUDA-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[NOT_0]] : (tensor<4x3xi1>) -> tensor<4x3xi32>
// CUDA-NEXT: return %[[CUSTOM_CALL_0]], %[[CONVERT_0]] : tensor<4x3x64x1xf64>, tensor<4x3xi32>
// CUDA-NEXT: }
