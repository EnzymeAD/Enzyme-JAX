// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=CUDA

func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
  %0:4 = enzymexla.lapack.gesvd %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
}

// CPU: func.func private @shlo_enzymexla_wrapper_lapack_sgesvd__wrapper_[[WRAPPER_ID:.*]](%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CPU-NEXT:   %c = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:   %0:5 = enzymexla.jit_call @enzymexla_wrapper_lapack_sgesvd_ (%c, %c, %arg0, %c, %cst_0, %cst, %c, %cst, %c, %c_1) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 9, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>, tensor<64x64xf32>)
// CPU-NEXT:   return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
// CPU-NEXT: }
// CPU-NEXT: llvm.func private @enzymexla_wrapper_lapack_sgesvd_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   %0 = llvm.mlir.constant(83 : i8) : i8
// CPU-NEXT:   %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:   %2 = llvm.mlir.constant(-1 : i64) : i64
// CPU-NEXT:   %3 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:   llvm.store %2, %3 : i64, !llvm.ptr
// CPU-NEXT:   %4 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr
// CPU-NEXT:   %5 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
// CPU-NEXT:   %6 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
// CPU-NEXT:   llvm.store %0, %5 : i8, !llvm.ptr
// CPU-NEXT:   llvm.store %0, %6 : i8, !llvm.ptr
// CPU-NEXT:   llvm.call @enzymexla_lapack_sgesvd_(%5, %6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %4, %3, %arg9, %1, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:   %7 = llvm.load %4 : !llvm.ptr -> f32
// CPU-NEXT:   %8 = llvm.fptosi %7 : f32 to i64
// CPU-NEXT:   %9 = llvm.alloca %8 x f32 : (i64) -> !llvm.ptr
// CPU-NEXT:   llvm.store %8, %3 : i64, !llvm.ptr
// CPU-NEXT:   llvm.call @enzymexla_lapack_sgesvd_(%5, %6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %9, %3, %arg9, %1, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func private @enzymexla_wrapper_lapack_sgesvd__batched(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   %c0_i64 = arith.constant 0 : i64
// CPU-NEXT:   %c12_i64 = arith.constant 12 : i64
// CPU-NEXT:   %c1_i64 = arith.constant 1 : i64
// CPU-NEXT:   scf.for %arg10 = %c0_i64 to %c12_i64 step %c1_i64  : i64 {
// CPU-NEXT:     %0 = llvm.getelementptr inbounds %arg2[%arg10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16384 x i8>
// CPU-NEXT:     %1 = llvm.getelementptr inbounds %arg4[%arg10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x i8>
// CPU-NEXT:     %2 = llvm.getelementptr inbounds %arg5[%arg10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16384 x i8>
// CPU-NEXT:     %3 = llvm.getelementptr inbounds %arg7[%arg10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16384 x i8>
// CPU-NEXT:     %4 = llvm.getelementptr inbounds %arg9[%arg10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
// CPU-NEXT:     llvm.call @enzymexla_wrapper_lapack_sgesvd_(%arg0, %arg1, %0, %arg3, %1, %2, %arg6, %3, %arg8, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:   }
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_lapack_sgesvd_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CPU-NEXT:   %0:4 = call @batched_shlo_enzymexla_wrapper_lapack_sgesvd__wrapper_0(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
// CPU-NEXT:   return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT: }
// CPU-NEXT: func.func private @batched_shlo_enzymexla_wrapper_lapack_sgesvd__wrapper_0(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CPU-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x3x64x64xf32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x3x64xf32>
// CPU-NEXT:   %c = stablehlo.constant dense<0> : tensor<4x3xi64>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:   %0:5 = enzymexla.jit_call @enzymexla_wrapper_lapack_sgesvd__batched (%c_1, %c_1, %arg0, %c_1, %cst_0, %cst, %c_1, %cst, %c_1, %c) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 9, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x3x64x64xf32>, tensor<i64>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<i64>, tensor<4x3x64x64xf32>, tensor<i64>, tensor<4x3xi64>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>, tensor<4x3x64x64xf32>)
// CPU-NEXT:   return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT: }

// CUDA: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CUDA-NEXT:   %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {api_version = 4 : i32, backend_config = {compute_uv = true, full_matrices = false, transposed = false}, operand_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi32>)
// CUDA-NEXT:   %1 = stablehlo.convert %0#4 : (tensor<4x3xi32>) -> tensor<4x3xi64>
// CUDA-NEXT:   return %0#2, %0#1, %0#3, %1 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CUDA-NEXT: }
