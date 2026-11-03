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
// CPU-NEXT: llvm.func private @enzymexla_lapack_dgtsv_wrapper_batched(%[[ARG0:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG2:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG3:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG4:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG5:[^:]+]]: !llvm.ptr {llvm.nofree}, %[[ARG6:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.readonly}, %[[ARG7:[^:]+]]: !llvm.ptr {llvm.nofree, llvm.writeonly}) {
// CPU-NEXT: %[[LB:.*]] = arith.constant 0 : i64
// CPU-NEXT: %[[UB:.*]] = arith.constant 12 : i64
// CPU-NEXT: %[[STEP:.*]] = arith.constant 1 : i64
// CPU-NEXT: scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] : i64 {
// CPU-NEXT: %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr inbounds %[[ARG2]][%[[IV]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i8>
// CPU-NEXT: %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr inbounds %[[ARG3]][%[[IV]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i8>
// CPU-NEXT: %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr inbounds %[[ARG4]][%[[IV]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i8>
// CPU-NEXT: %[[GETELEMENTPTR_5:.*]] = llvm.getelementptr inbounds %[[ARG5]][%[[IV]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i8>
// CPU-NEXT: %[[GETELEMENTPTR_7:.*]] = llvm.getelementptr inbounds %[[ARG7]][%[[IV]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
// CPU-NEXT: llvm.call @enzymexla_lapack_dgtsv_wrapper(%[[ARG0]], %[[ARG1]], %[[GETELEMENTPTR_2]], %[[GETELEMENTPTR_3]], %[[GETELEMENTPTR_4]], %[[GETELEMENTPTR_5]], %[[ARG6]], %[[GETELEMENTPTR_7]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT: }
// CPU-NEXT: llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_lapack_dgtsv_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU: func.func @main(%[[ARG0:[^:]+]]: tensor<4x3x64xf64>, %[[ARG1:[^:]+]]: tensor<4x3x64xf64>, %[[ARG2:[^:]+]]: tensor<4x3x64xf64>, %[[ARG3:[^:]+]]: tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi32>) {
// CPU-NEXT: %[[VAL_0:.*]]:2 = call @batched_enzymexla_lapack_dgtsv_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi64>)
// CPU-NEXT: %[[CONVERT_0:.*]] = stablehlo.convert %[[VAL_0]]#1 : (tensor<4x3xi64>) -> tensor<4x3xi32>
// CPU-NEXT: return %[[VAL_0]]#0, %[[CONVERT_0]] : tensor<4x3x64x1xf64>, tensor<4x3xi32>
// CPU-NEXT: }

// CPU: func.func private @batched_enzymexla_lapack_dgtsv_0(%[[ARG0:[^:]+]]: tensor<4x3x64xf64>, %[[ARG1:[^:]+]]: tensor<4x3x64xf64>, %[[ARG2:[^:]+]]: tensor<4x3x64xf64>, %[[ARG3:[^:]+]]: tensor<4x3x64x1xf64>) -> (tensor<4x3x64x1xf64>, tensor<4x3xi64>) {
// CPU-NEXT: %[[CONSTANT_2:.*]] = stablehlo.constant dense<-1> : tensor<4x3xi64>
// CPU-NEXT: %[[N:.*]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT: %[[NRHS:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT: %[[JIT_CALL_0:.*]]:5 = enzymexla.jit_call @enzymexla_lapack_dgtsv_wrapper_batched (%[[N]], %[[NRHS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[N]], %[[CONSTANT_2]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 7, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>, tensor<i64>, tensor<4x3xi64>) -> (tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64xf64>, tensor<4x3x64x1xf64>, tensor<4x3xi64>)
// CPU-NEXT: return %[[JIT_CALL_0]]#3, %[[JIT_CALL_0]]#4 : tensor<4x3x64x1xf64>, tensor<4x3xi64>
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
