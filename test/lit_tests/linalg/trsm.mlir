// RUN: enzymexlamlir-opt --lower-enzymexla-lapack={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU

func.func @main(%alpha: tensor<f32>, %a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = enzymexla.lapack.trsm %alpha, %a, %b {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>, transa = #enzymexla.transpose<none>, diag = #enzymexla.diag<nonunit>} : (tensor<f32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CPU-LABEL:  llvm.func @enzymexla_wrapper_lapack_strsm_
// CPU-LABEL:  llvm.func @enzymexla_lapack_strsm_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU:  func.func @main(%arg0: tensor<f32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CPU-NEXT:    %c = stablehlo.constant dense<76> : tensor<i8>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<85> : tensor<i8>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<78> : tensor<i8>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_wrapper_lapack_strsm_ (%c, %c_0, %c_1, %c_1, %c_2, %c_2, %arg0, %arg1, %c_2, %arg2, %c_2) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>, tensor<64x64xf32>, tensor<i64>) -> tensor<64x64xf32>
// CPU-NEXT:    return %0 : tensor<64x64xf32>
// CPU-NEXT:  }
