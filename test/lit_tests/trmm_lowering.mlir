// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cuda" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=tpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=TPU

func.func @trmm(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = enzymexla.blas.trmm %arg0, %arg1, %alpha {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transpose = #enzymexla.transpose<none>, diag = #enzymexla.diag<nonunit>} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<f32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
}

// CPU:  func.func private @enzymexla_blas_strmm_wrapper_0(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<f32>) -> tensor<64x32xf32> {
// CPU-NEXT:    %c = stablehlo.constant dense<82> : tensor<ui8>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<76> : tensor<ui8>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<78> : tensor<ui8>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %c_3 = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_blas_strmm_wrapper (%c, %c_0, %c_1, %c_1, %c_2, %c_3, %arg2, %arg0, %c_2, %arg1, %c_2) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>, tensor<64x32xf32>, tensor<i64>) -> tensor<64x32xf32>
// CPU-NEXT:    return %0 : tensor<64x32xf32>
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func private @enzymexla_blas_strmm_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    llvm.call @enzymexla_blas_strmm_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_blas_strmm_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT:  func.func @trmm(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
// CPU-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CPU-NEXT:    %0 = call @enzymexla_blas_strmm_wrapper_0(%arg0, %arg1, %cst) : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<f32>) -> tensor<64x32xf32>
// CPU-NEXT:    return %0 : tensor<64x32xf32>
// CPU-NEXT:  }

// CUDA:  func.func @trmm(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
// CUDA-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64x32xf32>
// CUDA-NEXT:    %0 = tensor.empty() : tensor<0xf32>
// CUDA-NEXT:    %1 = stablehlo.custom_call @enzymejax_cublas_trmm_ffi(%arg0, %arg1, %cst, %0) {api_version = 4 : i32, backend_config = {alpha_imag = 0.000000e+00 : f64, alpha_real = 2.000000e+00 : f64, diag = true, side = false, transpose = true, uplo = true, use_alpha_attribute = true}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<64x32xf32>, tensor<0xf32>) -> tensor<64x32xf32>
// CUDA-NEXT:    return %1 : tensor<64x32xf32>
// CUDA-NEXT:  }

// TPU:  func.func @trmm(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
// TPU-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x32xf32>
// TPU-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
// TPU-NEXT:    %1 = stablehlo.multiply %cst, %0 : tensor<64x32xf32>
// TPU-NEXT:    return %1 : tensor<64x32xf32>
// TPU-NEXT:  }