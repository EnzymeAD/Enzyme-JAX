// RUN: enzymexlamlir-opt --lower-blas-to-jit_call="backend=cpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-blas-to-stablehlo="symm=true" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=TPU

module {
    func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
        %alpha = stablehlo.constant dense<2.0> : tensor<f32>
        %beta = stablehlo.constant dense<3.0> : tensor<f32>
        %0 = blas.symm %alpha, %arg0, %arg1, %beta, %arg2 {side = #blas.side<left>, uplo = #blas.uplo<upper>} : (tensor<f32>, tensor<64x64xf32>, tensor<64x32xf32>, tensor<f32>, tensor<64x32xf32>) -> tensor<64x32xf32>
        return %0 : tensor<64x32xf32>
    }
}

// CPU: module {
// CPU-NEXT:   func.func private @enzymexla_blas_ssymm_wrapper_[[SYMMID:[0-9]+]](%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<64x32xf32> {
// CPU-DAG:     %[[cL:.+]] = stablehlo.constant dense<76> : tensor<ui8>
// CPU-DAG:     %[[cU:.+]] = stablehlo.constant dense<85> : tensor<ui8>
// CPU-DAG:     %[[c64:.+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-DAG:     %[[c32:.+]] = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT:     %0 = enzymexla.jit_call @enzymexla_blas_ssymm_wrapper (%[[cL]], %[[cU]], %[[c64]], %[[c32]], %arg3, %arg0, %[[c64]], %arg1, %[[c64]], %arg4, %arg2, %[[c64]]) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 10, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>, tensor<64x32xf32>, tensor<i64>, tensor<f32>, tensor<64x32xf32>, tensor<i64>) -> tensor<64x32xf32>
// CPU-NEXT:     return %0 : tensor<64x32xf32>
// CPU-NEXT:   }
// CPU-NEXT:   llvm.func private @enzymexla_blas_ssymm_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     llvm.call @enzymexla_blas_ssymm_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }
// CPU-NEXT:   llvm.func @enzymexla_blas_ssymm_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT:   func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
// CPU-NEXT:     %[[cst2:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CPU-NEXT:     %[[cst3:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CPU-NEXT:     %0 = call @enzymexla_blas_ssymm_wrapper_[[SYMMID:[0-9]+]](%arg0, %arg1, %arg2, %[[cst2]], %[[cst3]]) : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<64x32xf32>, tensor<f32>, tensor<f32>) -> tensor<64x32xf32>
// CPU-NEXT:     return %0 : tensor<64x32xf32>
// CPU-NEXT:   }
// CPU-NEXT: }

// TPU:  func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
// TPU-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<64x32xf32>
// TPU-NEXT:    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<64x32xf32>
// TPU-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// TPU-NEXT:    %1 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// TPU-NEXT:    %2 = stablehlo.compare  LT, %0, %1 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// TPU-NEXT:    %3 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// TPU-NEXT:    %4 = stablehlo.select %2, %arg0, %3 : tensor<64x64xi1>, tensor<64x64xf32>
// TPU-NEXT:    %5 = stablehlo.dot_general %4, %arg1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
// TPU-NEXT:    %6 = stablehlo.multiply %cst_0, %5 : tensor<64x32xf32>
// TPU-NEXT:    %7 = stablehlo.multiply %cst, %arg2 : tensor<64x32xf32>
// TPU-NEXT:    %8 = stablehlo.add %6, %7 : tensor<64x32xf32>
// TPU-NEXT:    return %8 : tensor<64x32xf32>
// TPU-NEXT:  }
