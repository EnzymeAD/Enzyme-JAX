// RUN: enzymexlamlir-opt --lower-enzymexla-lapack={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU

// Row-major uplo=U is the column-major lower factor: potrf('L') (char 76).

func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
  %0:2 = enzymexla.lapack.potrf %arg0 <{uplo = #enzymexla.uplo<U>}> : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
}

func.func @lower(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
  %0:2 = enzymexla.lapack.potrf %arg0 {uplo = #enzymexla.uplo<L>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
}

// CPU-LABEL:  llvm.func @enzymexla_wrapper_lapack_spotrf_
// CPU-LABEL:  llvm.func @enzymexla_lapack_spotrf_

// CPU:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<76> : tensor<i8>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %0:2 = enzymexla.jit_call @enzymexla_wrapper_lapack_spotrf_ (%c, %c_0, %arg0, %c_0, %c_1) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i8>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<i64>)
// CPU-NEXT:    return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
// CPU-NEXT:  }

// CPU:  func.func @lower(%[[A:.+]]: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
// CPU-NEXT:    %[[UPLO:.+]] = stablehlo.constant dense<76> : tensor<i8>
// CPU-NEXT:    %[[N:.+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %[[INFO:.+]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %[[R:.+]]:2 = enzymexla.jit_call @enzymexla_wrapper_lapack_spotrf_ (%[[UPLO]], %[[N]], %[[A]], %[[N]], %[[INFO]]) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>],
// CPU-SAME:      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>]
// CPU-NEXT:    return %[[R]]#0, %[[R]]#1 : tensor<64x64xf32>, tensor<i64>
// CPU-NEXT:  }
