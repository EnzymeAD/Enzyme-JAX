// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=tpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=TPU

module {
    func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
        %alpha = stablehlo.constant dense<2.0> : tensor<f32>
        %beta = stablehlo.constant dense<3.0> : tensor<f32>
        %0 = enzymexla.blas.symm %arg0, %arg1, %arg2, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<64x32xf32>, tensor<f32>, tensor<f32>) -> tensor<64x32xf32>
        return %0 : tensor<64x32xf32>
    }
}

// CPU: module {
// CPU-NEXT:   func.func private @enzymexla_blas_ssymm_wrapper_0(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<64x32xf32> {
// CPU-NEXT:     %c = stablehlo.constant dense<76> : tensor<ui8>
// CPU-NEXT:     %c_0 = stablehlo.constant dense<85> : tensor<ui8>
// CPU-NEXT:     %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:     %c_2 = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT:     %0 = enzymexla.jit_call @enzymexla_blas_ssymm_wrapper (%c, %c_0, %c_1, %c_2, %arg3, %arg0, %c_1, %arg1, %c_1, %arg4, %arg2, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 10, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>, tensor<64x32xf32>, tensor<i64>, tensor<f32>, tensor<64x32xf32>, tensor<i64>) -> tensor<64x32xf32>
// CPU-NEXT:     return %0 : tensor<64x32xf32>
// CPU-NEXT:   }
// CPU-NEXT:   llvm.func private @enzymexla_blas_ssymm_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     llvm.call @enzymexla_blas_ssymm_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }
// CPU-NEXT:   llvm.func @enzymexla_blas_ssymm_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT:   func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
// CPU-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CPU-NEXT:     %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CPU-NEXT:     %0 = call @enzymexla_blas_ssymm_wrapper_0(%arg0, %arg1, %arg2, %cst, %cst_0) : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<64x32xf32>, tensor<f32>, tensor<f32>) -> tensor<64x32xf32>
// CPU-NEXT:     return %0 : tensor<64x32xf32>
// CPU-NEXT:   }
// CPU-NEXT: }

// TPU: module {
// TPU-NEXT:   func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
// TPU-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<64x32xf32>
// TPU-NEXT:     %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<64x32xf32>
// TPU-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
// TPU-NEXT:     %1 = stablehlo.multiply %cst_0, %arg2 : tensor<64x32xf32>
// TPU-NEXT:     %2 = stablehlo.multiply %cst, %0 : tensor<64x32xf32>
// TPU-NEXT:     %3 = stablehlo.add %2, %1 : tensor<64x32xf32>
// TPU-NEXT:     return %3 : tensor<64x32xf32>
// TPU-NEXT:   }
// TPU-NEXT: }