// RUN: enzymexlamlir-opt --lower-enzymexla-lapack={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
  %0:2 = enzymexla.lapack.potrf %arg0 {uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
}

// CPU:  llvm.func @enzymexla_wrapper_lapack_spotrf_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzymexla_lapack_spotrf_(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapack_spotrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<i64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<76> : tensor<i8>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %0:2 = enzymexla.jit_call @enzymexla_wrapper_lapack_spotrf_ (%c, %c_0, %arg0, %c_0, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<i8>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<i64>)
// CPU-NEXT:    return %0#0, %0#1 : tensor<64x64xf32>, tensor<i64>
// CPU-NEXT:  }
