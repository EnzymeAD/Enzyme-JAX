// RUN: enzymexlamlir-opt --lower-lapack-to-jit_call={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
  %0:2 = lapack.potrf %arg0 {uplo = #blas.uplo<upper>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
}

// CPU-LABEL:  llvm.func @enzymexla_wrapper_lapack_spotrf_
// CPU-LABEL:  llvm.func @enzymexla_lapack_spotrf_

// CPU:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<85> : tensor<i8>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %0:2 = enzymexla.jit_call @enzymexla_wrapper_lapack_spotrf_ (%c, %c_0, %arg0, %c_0, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<i8>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<i64>)
// CPU-NEXT:    return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
// CPU-NEXT:  }
