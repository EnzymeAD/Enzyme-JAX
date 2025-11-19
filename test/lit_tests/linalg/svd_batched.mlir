// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},lower-enzymexla-lapack{backend=cuda blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},lower-enzymexla-lapack{backend=tpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
    %0:4 = enzymexla.linalg.svd %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
  }
}

// CPU: llvm.func private @enzymexla_wrapper_lapack_sgesdd_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(8 : i64) : i64
// CPU-NEXT:     %1 = llvm.mlir.constant(83 : i8) : i8
// CPU-NEXT:     %2 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %3 = llvm.mlir.constant(-1 : i64) : i64
// CPU-NEXT:     %4 = llvm.alloca %2 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %3, %4 : i64, !llvm.ptr
// CPU-NEXT:     %5 = llvm.alloca %2 x f32 : (i64) -> !llvm.ptr
// CPU-NEXT:     %6 = llvm.alloca %2 x i8 : (i64) -> !llvm.ptr
// CPU-NEXT:     %7 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:     %8 = llvm.load %arg1 : !llvm.ptr -> i64
// CPU-NEXT:     %9 = arith.minsi %7, %8 : i64
// CPU-NEXT:     %10 = llvm.mul %9, %0 : i64
// CPU-NEXT:     %11 = llvm.alloca %10 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %1, %6 : i8, !llvm.ptr
// CPU-NEXT:     llvm.call @enzymexla_lapack_sgesdd_(%6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %5, %4, %11, %arg9, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
// CPU-NEXT:     %12 = llvm.load %5 : !llvm.ptr -> f32
// CPU-NEXT:     %13 = llvm.fptosi %12 : f32 to i64
// CPU-NEXT:     %14 = llvm.alloca %13 x f32 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %13, %4 : i64, !llvm.ptr
// CPU-NEXT:     llvm.call @enzymexla_lapack_sgesdd_(%6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %14, %4, %11, %arg9, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }
// CPU-NEXT:   llvm.func @enzymexla_lapack_sgesdd_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
// CPU-NEXT:   func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CPU-NEXT:     %0:4 = call @batched_shlo_enzymexla_wrapper_lapack_sgesdd__wrapper_0(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
// CPU-NEXT:     return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT:   }
// CPU-NEXT:   func.func private @batched_shlo_enzymexla_wrapper_lapack_sgesdd__wrapper_0(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CPU-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
// CPU-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
// CPU-NEXT:     %c = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:     %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CPU-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:     %c_3 = stablehlo.constant dense<12> : tensor<i64>
// CPU-NEXT:     %cst_4 = arith.constant dense<0> : tensor<4x3xi64>
// CPU-NEXT:     %cst_5 = arith.constant dense<0.000000e+00> : tensor<4x3x64xf32>
// CPU-NEXT:     %cst_6 = arith.constant dense<0.000000e+00> : tensor<4x3x64x64xf32>
// CPU-NEXT:     %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:     %0:5 = stablehlo.while(%iterArg = %c_7, %iterArg_8 = %cst_6, %iterArg_9 = %cst_5, %iterArg_10 = %cst_6, %iterArg_11 = %cst_4) : tensor<i64>, tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT:     cond {
// CPU-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT:       stablehlo.return %1 : tensor<i1>
// CPU-NEXT:     } do {
// CPU-NEXT:       %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CPU-NEXT:       %2 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
// CPU-NEXT:       %3 = stablehlo.divide %iterArg, %c_1 : tensor<i64>
// CPU-NEXT:       %4 = stablehlo.dynamic_slice %arg0, %2, %3, %c_7, %c_7, sizes = [1, 1, 64, 64] : (tensor<4x3x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x64xf32>
// CPU-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
// CPU-NEXT:       %6:5 = enzymexla.jit_call @enzymexla_wrapper_lapack_sgesdd_ (%c, %c, %5, %c, %cst_0, %cst, %c, %cst, %c, %c_7) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 9, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<i64>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>, tensor<64x64xf32>)
// CPU-NEXT:       %7 = stablehlo.reshape %6#0 : (tensor<64x64xf32>) -> tensor<1x1x64x64xf32>
// CPU-NEXT:       %8 = stablehlo.dynamic_update_slice %iterArg_8, %7, %2, %3, %c_7, %c_7 : (tensor<4x3x64x64xf32>, tensor<1x1x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64x64xf32>
// CPU-NEXT:       %9 = stablehlo.reshape %6#1 : (tensor<64xf32>) -> tensor<1x1x64xf32>
// CPU-NEXT:       %10 = stablehlo.dynamic_update_slice %iterArg_9, %9, %2, %3, %c_7 : (tensor<4x3x64xf32>, tensor<1x1x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64xf32>
// CPU-NEXT:       %11 = stablehlo.reshape %6#2 : (tensor<64x64xf32>) -> tensor<1x1x64x64xf32>
// CPU-NEXT:       %12 = stablehlo.dynamic_update_slice %iterArg_10, %11, %2, %3, %c_7, %c_7 : (tensor<4x3x64x64xf32>, tensor<1x1x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x64x64xf32>
// CPU-NEXT:       %13 = stablehlo.reshape %6#3 : (tensor<i64>) -> tensor<1x1xi64>
// CPU-NEXT:       %14 = stablehlo.dynamic_update_slice %iterArg_11, %13, %2, %3 : (tensor<4x3xi64>, tensor<1x1xi64>, tensor<i64>, tensor<i64>) -> tensor<4x3xi64>
// CPU-NEXT:       stablehlo.return %1, %8, %10, %12, %14 : tensor<i64>, tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT:     }
// CPU-NEXT:     return %0#1, %0#2, %0#3, %0#4 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT:   }

// CUDA: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CUDA-NEXT:     %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {api_version = 4 : i32, backend_config = {compute_uv = true, full_matrices = false}, operand_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi32>)
// CUDA-NEXT:     %1 = stablehlo.convert %0#4 : (tensor<4x3xi32>) -> tensor<4x3xi64>
// CUDA-NEXT:     %2 = stablehlo.transpose %0#3, dims = [0, 1, 3, 2] : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xf32>
// CUDA-NEXT:     return %0#2, %0#1, %2, %1 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CUDA-NEXT:   }

// TPU: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// TPU-NEXT:   %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:   %0:3 = stablehlo.custom_call @SVD(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>)
// TPU-NEXT:   %1 = stablehlo.transpose %0#2, dims = [0, 1, 3, 2] : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xf32>
// TPU-NEXT:   %2 = stablehlo.is_finite %0#0 : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xi1>
// TPU-NEXT:   %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.and across dimensions = [2, 3] : (tensor<4x3x64x64xi1>, tensor<i1>) -> tensor<4x3xi1>
// TPU-NEXT:   %4 = stablehlo.not %3 : tensor<4x3xi1>
// TPU-NEXT:   %5 = stablehlo.convert %4 : (tensor<4x3xi1>) -> tensor<4x3xi64>
// TPU-NEXT:   return %0#0, %0#1, %1, %5 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// TPU-NEXT: }
