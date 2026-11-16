// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=cpu blas_int_width=64" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=cuda" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=tpu" %s | FileCheck %s --check-prefix=TPU

func.func @main(%dl: tensor<64xf32>, %d: tensor<64xf32>, %du: tensor<64xf32>, %b: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
  %0:2 = enzymexla.lapack.gtsv %dl, %d, %du, %b : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>)
  return %0#0, %0#1 : tensor<64x32xf32>, tensor<i32>
}

// CPU: func.func private @enzymexla_lapack_sgtsv_0(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i64>) {
// CPU-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT: %[[CONSTANT_2:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT: %[[JIT_CALL_0:.*]]:5 = enzymexla.jit_call @enzymexla_lapack_sgtsv_wrapper (%[[CONSTANT_0]], %[[CONSTANT_1]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[CONSTANT_0]], %[[CONSTANT_2]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 7, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>, tensor<i64>)
// CPU-NEXT: return %[[JIT_CALL_0]]#3, %[[JIT_CALL_0]]#4 : tensor<64x32xf32>, tensor<i64>
// CPU-NEXT: }

// CPU: llvm.func private @enzymexla_lapack_sgtsv_wrapper(%[[ARG0:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG2:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG3:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG4:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG5:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG6:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG7:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.writeonly}) {
// CPU-NEXT: %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[ARG2]][1] : (!llvm.ptr) -> !llvm.ptr, f32
// CPU-NEXT: llvm.call @enzymexla_lapack_sgtsv_(%[[ARG0]], %[[ARG1]], %[[GETELEMENTPTR_0]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT: llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_lapack_sgtsv_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU: func.func @main(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
// CPU-NEXT: %[[VAL_0:.*]]:2 = call @enzymexla_lapack_sgtsv_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i64>)
// CPU-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[VAL_0]]#1 : (tensor<i64>) -> tensor<i32>
// CPU-NEXT: return %[[VAL_0]]#0, %[[CONVERT_0]] : tensor<64x32xf32>, tensor<i32>
// CPU-NEXT: }

// CUDA: func.func @main(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
// CUDA-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CUDA-NEXT: %[[CUSTOM_CALL_0:.*]] = stablehlo.custom_call @cusparse_gtsv2_ffi(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) {api_version = 4 : i32, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
// CUDA-NEXT: %[[REAL_0:.*]] = stablehlo.real %[[CUSTOM_CALL_0]] : tensor<64x32xf32>
// CUDA-NEXT: %[[IS_FINITE_0:.*]] = stablehlo.is_finite %[[REAL_0]] : (tensor<64x32xf32>) -> tensor<64x32xi1>
// CUDA-NEXT: %[[IMAG_0:.*]] = stablehlo.imag %[[CUSTOM_CALL_0]] : tensor<64x32xf32>
// CUDA-NEXT: %[[IS_FINITE_1:.*]] = stablehlo.is_finite %[[IMAG_0]] : (tensor<64x32xf32>) -> tensor<64x32xi1>
// CUDA-NEXT: %[[AND_0:.*]] = stablehlo.and %[[IS_FINITE_0]], %[[IS_FINITE_1]] : tensor<64x32xi1>
// CUDA-NEXT: %[[REDUCE_0:.*]] = stablehlo.reduce(%[[AND_0]] init: %[[CONSTANT_0]]) applies stablehlo.and across dimensions = [0, 1] : (tensor<64x32xi1>, tensor<i1>) -> tensor<i1>
// CUDA-NEXT: %[[NOT_0:.*]] = stablehlo.not %[[REDUCE_0]] : tensor<i1>
// CUDA-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[NOT_0]] : (tensor<i1>) -> tensor<i32>
// CUDA-NEXT: return %[[CUSTOM_CALL_0]], %[[CONVERT_0]] : tensor<64x32xf32>, tensor<i32>
// CUDA-NEXT: }

// TPU: func.func @main(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
// TPU-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<63> : tensor<i64>
// TPU-NEXT: %[[CONSTANT_2:.*]] = stablehlo.constant dense<64> : tensor<i64>
// TPU-NEXT: %[[CONSTANT_3:.*]] = stablehlo.constant dense<1> : tensor<i64>
// TPU-NEXT: %[[CONSTANT_4:.*]] = stablehlo.constant dense<0> : tensor<i64>
// TPU-NEXT: %[[BROADCAST_IN_DIM_0:.*]] = stablehlo.broadcast_in_dim %[[ARG0]], dims = [0] : (tensor<64xf32>) -> tensor<64x32xf32>
// TPU-NEXT: %[[BROADCAST_IN_DIM_1:.*]] = stablehlo.broadcast_in_dim %[[ARG1]], dims = [0] : (tensor<64xf32>) -> tensor<64x32xf32>
// TPU-NEXT: %[[BROADCAST_IN_DIM_2:.*]] = stablehlo.broadcast_in_dim %[[ARG2]], dims = [0] : (tensor<64xf32>) -> tensor<64x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_0:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_1]], %[[CONSTANT_4]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_1:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_2]], %[[CONSTANT_4]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DIVIDE_0:.*]] = stablehlo.divide %[[DYNAMIC_SLICE_1]], %[[DYNAMIC_SLICE_0]] : tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_0:.*]] = stablehlo.dynamic_update_slice %[[ARG3]], %[[DIVIDE_0]], %[[CONSTANT_4]], %[[CONSTANT_4]] : (tensor<64x32xf32>, tensor<1x32xf32>, tensor<i64>, tensor<i64>) -> tensor<64x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_2:.*]] = stablehlo.dynamic_slice %[[ARG3]], %[[CONSTANT_4]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DIVIDE_1:.*]] = stablehlo.divide %[[DYNAMIC_SLICE_2]], %[[DYNAMIC_SLICE_0]] : tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_1:.*]] = stablehlo.dynamic_update_slice %[[ARG3]], %[[DIVIDE_1]], %[[CONSTANT_4]], %[[CONSTANT_4]] : (tensor<64x32xf32>, tensor<1x32xf32>, tensor<i64>, tensor<i64>) -> tensor<64x32xf32>
// TPU-NEXT: %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_0:.*]] = %[[CONSTANT_3]], %[[VAL_1:.*]] = %[[DYNAMIC_UPDATE_SLICE_0]], %[[VAL_2:.*]] = %[[DYNAMIC_UPDATE_SLICE_1]]) : tensor<i64>, tensor<64x32xf32>, tensor<64x32xf32>
// TPU-NEXT: cond {
// TPU-NEXT: %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_0]], %[[CONSTANT_2]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// TPU-NEXT: stablehlo.return %[[COMPARE_0]] : tensor<i1>
// TPU-NEXT: } do {
// TPU-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[VAL_0]], %[[CONSTANT_3]] : tensor<i64>
// TPU-NEXT: %[[SUBTRACT_0:.*]] = stablehlo.subtract %[[VAL_0]], %[[CONSTANT_3]] : tensor<i64>
// TPU-NEXT: %[[DYNAMIC_SLICE_3:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_0]], %[[VAL_0]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_4:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_1]], %[[VAL_0]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_5:.*]] = stablehlo.dynamic_slice %[[VAL_1]], %[[SUBTRACT_0]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[MULTIPLY_0:.*]] = stablehlo.multiply %[[DYNAMIC_SLICE_3]], %[[DYNAMIC_SLICE_5]] : tensor<1x32xf32>
// TPU-NEXT: %[[SUBTRACT_1:.*]] = stablehlo.subtract %[[DYNAMIC_SLICE_4]], %[[MULTIPLY_0]] : tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_6:.*]] = stablehlo.dynamic_slice %[[BROADCAST_IN_DIM_2]], %[[VAL_0]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DIVIDE_2:.*]] = stablehlo.divide %[[DYNAMIC_SLICE_6]], %[[SUBTRACT_1]] : tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_7:.*]] = stablehlo.dynamic_slice %[[ARG3]], %[[VAL_0]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_8:.*]] = stablehlo.dynamic_slice %[[VAL_2]], %[[SUBTRACT_0]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[MULTIPLY_1:.*]] = stablehlo.multiply %[[DYNAMIC_SLICE_3]], %[[DYNAMIC_SLICE_8]] : tensor<1x32xf32>
// TPU-NEXT: %[[SUBTRACT_2:.*]] = stablehlo.subtract %[[DYNAMIC_SLICE_7]], %[[MULTIPLY_1]] : tensor<1x32xf32>
// TPU-NEXT: %[[DIVIDE_3:.*]] = stablehlo.divide %[[SUBTRACT_2]], %[[SUBTRACT_1]] : tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_2:.*]] = stablehlo.dynamic_update_slice %[[VAL_1]], %[[DIVIDE_2]], %[[VAL_0]], %[[CONSTANT_4]] : (tensor<64x32xf32>, tensor<1x32xf32>, tensor<i64>, tensor<i64>) -> tensor<64x32xf32>
// TPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_3:.*]] = stablehlo.dynamic_update_slice %[[VAL_2]], %[[DIVIDE_3]], %[[VAL_0]], %[[CONSTANT_4]] : (tensor<64x32xf32>, tensor<1x32xf32>, tensor<i64>, tensor<i64>) -> tensor<64x32xf32>
// TPU-NEXT: stablehlo.return %[[ADD_0]], %[[DYNAMIC_UPDATE_SLICE_2]], %[[DYNAMIC_UPDATE_SLICE_3]] : tensor<i64>, tensor<64x32xf32>, tensor<64x32xf32>
// TPU-NEXT: }
// TPU-NEXT: %[[WHILE_1:.*]]:2 = stablehlo.while(%[[VAL_3:.*]] = %[[CONSTANT_3]], %[[VAL_4:.*]] = %[[WHILE_0]]#2) : tensor<i64>, tensor<64x32xf32>
// TPU-NEXT: cond {
// TPU-NEXT: %[[COMPARE_1:.*]] = stablehlo.compare LT, %[[VAL_3]], %[[CONSTANT_2]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// TPU-NEXT: stablehlo.return %[[COMPARE_1]] : tensor<i1>
// TPU-NEXT: } do {
// TPU-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[VAL_3]], %[[CONSTANT_3]] : tensor<i64>
// TPU-NEXT: %[[SUBTRACT_3:.*]] = stablehlo.subtract %[[CONSTANT_1]], %[[VAL_3]] : tensor<i64>
// TPU-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[SUBTRACT_3]], %[[CONSTANT_3]] : tensor<i64>
// TPU-NEXT: %[[DYNAMIC_SLICE_9:.*]] = stablehlo.dynamic_slice %[[WHILE_0]]#2, %[[SUBTRACT_3]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_10:.*]] = stablehlo.dynamic_slice %[[WHILE_0]]#1, %[[SUBTRACT_3]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_SLICE_11:.*]] = stablehlo.dynamic_slice %[[VAL_4]], %[[ADD_2]], %[[CONSTANT_4]], sizes = [1, 32] : (tensor<64x32xf32>, tensor<i64>, tensor<i64>) -> tensor<1x32xf32>
// TPU-NEXT: %[[MULTIPLY_2:.*]] = stablehlo.multiply %[[DYNAMIC_SLICE_10]], %[[DYNAMIC_SLICE_11]] : tensor<1x32xf32>
// TPU-NEXT: %[[SUBTRACT_4:.*]] = stablehlo.subtract %[[DYNAMIC_SLICE_9]], %[[MULTIPLY_2]] : tensor<1x32xf32>
// TPU-NEXT: %[[DYNAMIC_UPDATE_SLICE_4:.*]] = stablehlo.dynamic_update_slice %[[VAL_4]], %[[SUBTRACT_4]], %[[SUBTRACT_3]], %[[CONSTANT_4]] : (tensor<64x32xf32>, tensor<1x32xf32>, tensor<i64>, tensor<i64>) -> tensor<64x32xf32>
// TPU-NEXT: stablehlo.return %[[ADD_1]], %[[DYNAMIC_UPDATE_SLICE_4]] : tensor<i64>, tensor<64x32xf32>
// TPU-NEXT: }
// TPU-NEXT: %[[REAL_0:.*]] = stablehlo.real %[[WHILE_1]]#1 : tensor<64x32xf32>
// TPU-NEXT: %[[IS_FINITE_0:.*]] = stablehlo.is_finite %[[REAL_0]] : (tensor<64x32xf32>) -> tensor<64x32xi1>
// TPU-NEXT: %[[IMAG_0:.*]] = stablehlo.imag %[[WHILE_1]]#1 : tensor<64x32xf32>
// TPU-NEXT: %[[IS_FINITE_1:.*]] = stablehlo.is_finite %[[IMAG_0]] : (tensor<64x32xf32>) -> tensor<64x32xi1>
// TPU-NEXT: %[[AND_0:.*]] = stablehlo.and %[[IS_FINITE_0]], %[[IS_FINITE_1]] : tensor<64x32xi1>
// TPU-NEXT: %[[REDUCE_0:.*]] = stablehlo.reduce(%[[AND_0]] init: %[[CONSTANT_0]]) applies stablehlo.and across dimensions = [0, 1] : (tensor<64x32xi1>, tensor<i1>) -> tensor<i1>
// TPU-NEXT: %[[NOT_0:.*]] = stablehlo.not %[[REDUCE_0]] : tensor<i1>
// TPU-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[NOT_0]] : (tensor<i1>) -> tensor<i32>
// TPU-NEXT: return %[[WHILE_1]]#1, %[[CONVERT_0]] : tensor<64x32xf32>, tensor<i32>
// TPU-NEXT: }
