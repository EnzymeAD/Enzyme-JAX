// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=cuda" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --lower-enzymexla-blas="backend=tpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=TPU

module {
    func.func @main1(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %alpha = stablehlo.constant dense<2.0> : tensor<f32>
        %beta = stablehlo.constant dense<3.0> : tensor<f32>
        %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {fill, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
        return %0 : tensor<64x64xf32>
    }
}

// CPU: func.func private @enzymexla_blas_ssyrk_wrapper_[[SYRKID:[0-9]+]](%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<64x64xf32> {
// CPU-NEXT:   %c = stablehlo.constant dense<78> : tensor<ui8>
// CPU-NEXT:   %c_0 = stablehlo.constant dense<85> : tensor<ui8>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:   %c_2 = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT:   %0 = enzymexla.jit_call @enzymexla_blas_ssyrk_wrapper (%c_0, %c, %c_1, %c_2, %arg2, %arg0, %c_1, %arg3, %arg1, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 8, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x32xf32>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>) -> tensor<64x64xf32>
// CPU-NEXT:   return %0 : tensor<64x64xf32>
// CPU-NEXT: }
// CPU-NEXT: llvm.func private @enzymexla_blas_ssyrk_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:   llvm.call @enzymexla_blas_ssyrk_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_blas_ssyrk_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT: func.func @main1(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CPU-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CPU-NEXT:   %0 = call @enzymexla_blas_ssyrk_wrapper_[[SYRKID]](%arg0, %arg1, %cst, %cst_0) {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CPU-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// CPU-NEXT:   %2 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// CPU-NEXT:   %3 = stablehlo.compare  LT, %1, %2 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// CPU-NEXT:   %4 = stablehlo.transpose %0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CPU-NEXT:   %5 = stablehlo.select %3, %0, %4 : tensor<64x64xi1>, tensor<64x64xf32>
// CPU-NEXT:   return %5 : tensor<64x64xf32>
// CPU-NEXT: }

// CUDA: func.func @main1(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CUDA-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CUDA-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CUDA-NEXT:   %0 = stablehlo.custom_call @cusolver_syrk_ffi(%arg0, %arg1, %cst, %cst_0) {api_version = 4 : i32, backend_config = {transpose = true}, enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>], operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CUDA-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// CUDA-NEXT:   %2 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// CUDA-NEXT:   %3 = stablehlo.compare  LT, %1, %2 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// CUDA-NEXT:   %4 = stablehlo.transpose %0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CUDA-NEXT:   %5 = stablehlo.select %3, %0, %4 : tensor<64x64xi1>, tensor<64x64xf32>
// CUDA-NEXT:   return %5 : tensor<64x64xf32>
// CUDA-NEXT: }

// TPU: func.func @main1(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// TPU-NEXT:     %cst = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// TPU-NEXT:     %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<64x64xf32>
// TPU-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// TPU-NEXT:     %1 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// TPU-NEXT:     %2 = stablehlo.compare  LT, %0, %1 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// TPU-NEXT:     %3 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// TPU-NEXT:     %4 = stablehlo.select %2, %arg1, %3 : tensor<64x64xi1>, tensor<64x64xf32>
// TPU-NEXT:     %5 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
// TPU-NEXT:     %6 = stablehlo.multiply %cst_0, %5 : tensor<64x64xf32>
// TPU-NEXT:     %7 = stablehlo.multiply %cst, %4 : tensor<64x64xf32>
// TPU-NEXT:     %8 = stablehlo.add %6, %7 : tensor<64x64xf32>
// TPU-NEXT:     return %8 : tensor<64x64xf32>
// TPU-NEXT: }

module {
    func.func @main2(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %alpha = stablehlo.constant dense<2.0> : tensor<f32>
        %beta = stablehlo.constant dense<3.0> : tensor<f32>
        %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<L>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
        return %0 : tensor<64x64xf32>
    }
}

// CPU: func.func private @enzymexla_blas_ssyrk_wrapper_[[SYRKID:[0-9]+]](%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<64x64xf32> {
// CPU-NEXT:   %c = stablehlo.constant dense<78> : tensor<ui8>
// CPU-NEXT:   %c_0 = stablehlo.constant dense<76> : tensor<ui8>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:   %c_2 = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT:   %0 = enzymexla.jit_call @enzymexla_blas_ssyrk_wrapper (%c_0, %c, %c_1, %c_2, %arg2, %arg0, %c_1, %arg3, %arg1, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 8, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x32xf32>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>) -> tensor<64x64xf32>
// CPU-NEXT:   return %0 : tensor<64x64xf32>
// CPU-NEXT: }
// CPU-NEXT: llvm.func private @enzymexla_blas_ssyrk_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:   llvm.call @enzymexla_blas_ssyrk_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_blas_ssyrk_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT: func.func @main2(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CPU-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CPU-NEXT:   %0 = call @enzymexla_blas_ssyrk_wrapper_[[SYRKID]](%arg0, %arg1, %cst, %cst_0) : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CPU-NEXT:   return %0 : tensor<64x64xf32>
// CPU-NEXT: }

// CUDA: func.func @main2(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CUDA-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CUDA-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CUDA-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// CUDA-NEXT:   %1 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// CUDA-NEXT:   %2 = stablehlo.compare  GT, %0, %1 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// CUDA-NEXT:   %3 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CUDA-NEXT:   %4 = stablehlo.select %2, %arg1, %3 : tensor<64x64xi1>, tensor<64x64xf32>
// CUDA-NEXT:   %5 = stablehlo.custom_call @cusolver_syrk_ffi(%arg0, %4, %cst, %cst_0) {api_version = 4 : i32, backend_config = {transpose = true}, enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>], operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CUDA-NEXT:   %6 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// CUDA-NEXT:   %7 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// CUDA-NEXT:   %8 = stablehlo.compare  LT, %6, %7 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// CUDA-NEXT:   %9 = stablehlo.transpose %5, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CUDA-NEXT:   %10 = stablehlo.select %8, %5, %9 : tensor<64x64xi1>, tensor<64x64xf32>
// CUDA-NEXT:   return %10 : tensor<64x64xf32>
// CUDA-NEXT: }

// TPU: func.func @main2(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// TPU-NEXT:     %cst = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// TPU-NEXT:     %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<64x64xf32>
// TPU-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<64x64xi32>
// TPU-NEXT:     %1 = stablehlo.iota dim = 1 : tensor<64x64xi32>
// TPU-NEXT:     %2 = stablehlo.compare  GT, %0, %1 : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
// TPU-NEXT:     %3 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// TPU-NEXT:     %4 = stablehlo.select %2, %arg1, %3 : tensor<64x64xi1>, tensor<64x64xf32>
// TPU-NEXT:     %5 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
// TPU-NEXT:     %6 = stablehlo.multiply %cst_0, %5 : tensor<64x64xf32>
// TPU-NEXT:     %7 = stablehlo.multiply %cst, %4 : tensor<64x64xf32>
// TPU-NEXT:     %8 = stablehlo.add %6, %7 : tensor<64x64xf32>
// TPU-NEXT:     return %8 : tensor<64x64xf32>
// TPU-NEXT: }

module {
    func.func @main3(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %alpha = stablehlo.constant dense<2.0> : tensor<f32>
        %beta = stablehlo.constant dense<3.0> : tensor<f32>
        %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<F>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
        return %0 : tensor<64x64xf32>
    }
}

// CPU: func.func private @enzymexla_blas_ssyrk_wrapper_[[SYRKID:[0-9]+]](%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<64x64xf32> {
// CPU-NEXT:   %c = stablehlo.constant dense<78> : tensor<ui8>
// CPU-NEXT:   %c_0 = stablehlo.constant dense<85> : tensor<ui8>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:   %c_2 = stablehlo.constant dense<32> : tensor<i64>
// CPU-NEXT:   %0 = enzymexla.jit_call @enzymexla_blas_ssyrk_wrapper (%c_0, %c, %c_1, %c_2, %arg2, %arg0, %c_1, %arg3, %arg1, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 8, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<64x32xf32>, tensor<i64>, tensor<f32>, tensor<64x64xf32>, tensor<i64>) -> tensor<64x64xf32>
// CPU-NEXT:   return %0 : tensor<64x64xf32>
// CPU-NEXT: }
// CPU-NEXT: llvm.func private @enzymexla_blas_ssyrk_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:   llvm.call @enzymexla_blas_ssyrk_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_blas_ssyrk_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT: func.func @main3(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CPU-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CPU-NEXT:   %0 = call @enzymexla_blas_ssyrk_wrapper_[[SYRKID]](%arg0, %arg1, %cst, %cst_0) : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CPU-NEXT:   return %0 : tensor<64x64xf32>
// CPU-NEXT: }

// CUDA: func.func @main3(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CUDA-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CUDA-NEXT:     %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CUDA-NEXT:     %0 = stablehlo.custom_call @cusolver_syrk_ffi(%arg0, %arg1, %cst, %cst_0) {api_version = 4 : i32, backend_config = {transpose = true}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CUDA-NEXT:     return %0 : tensor<64x64xf32>
// CUDA-NEXT: }

// TPU: func.func @main3(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
// TPU-NEXT:     %cst = stablehlo.constant dense<3.000000e+00> : tensor<64x64xf32>
// TPU-NEXT:     %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<64x64xf32>
// TPU-NEXT:     %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
// TPU-NEXT:     %1 = stablehlo.multiply %cst_0, %0 : tensor<64x64xf32>
// TPU-NEXT:     %2 = stablehlo.multiply %cst, %arg1 : tensor<64x64xf32>
// TPU-NEXT:     %3 = stablehlo.add %1, %2 : tensor<64x64xf32>
// TPU-NEXT:     return %3 : tensor<64x64xf32>
// TPU-NEXT: }

module {
    func.func @main4(%arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x4xf32>
        %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
        %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %0 = enzymexla.blas.syrk %arg0, %cst_0, %cst_1, %cst_2 {fill, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<U>} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
        %1 = stablehlo.multiply %cst, %arg1 : tensor<4x4xf32>
        %2 = stablehlo.add %0, %1 : tensor<4x4xf32>
        return %2 : tensor<4x4xf32>
    }
}

// CPU: func.func private @enzymexla_blas_ssyrk_wrapper_[[SYRKID:[0-9]+]](%arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<4x4xf32> {
// CPU-NEXT:   %c = stablehlo.constant dense<84> : tensor<ui8>
// CPU-NEXT:   %c_0 = stablehlo.constant dense<85> : tensor<ui8>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CPU-NEXT:   %c_2 = stablehlo.constant dense<5> : tensor<i64>
// CPU-NEXT:   %0 = enzymexla.jit_call @enzymexla_blas_ssyrk_wrapper (%c_0, %c, %c_1, %c_2, %arg2, %arg0, %c_2, %arg3, %arg1, %c_1) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 8, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f32>, tensor<5x4xf32>, tensor<i64>, tensor<f32>, tensor<4x4xf32>, tensor<i64>) -> tensor<4x4xf32>
// CPU-NEXT:   return %0 : tensor<4x4xf32>
// CPU-NEXT: }
// CPU-NEXT: llvm.func private @enzymexla_blas_ssyrk_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:   llvm.call @enzymexla_blas_ssyrk_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT: llvm.func @enzymexla_blas_ssyrk_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
// CPU-NEXT: func.func @main4(%arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CPU-NEXT{LITERAL}:   %c = stablehlo.constant dense<[[false, true, true, true], [false, false, true, true], [false, false, false, true], [false, false, false, false]]> : tensor<4x4xi1>
// CPU-NEXT:   %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x4xf32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// CPU-NEXT:   %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CPU-NEXT:   %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CPU-NEXT:   %0 = call @enzymexla_blas_ssyrk_wrapper_[[SYRKID]](%arg0, %cst_0, %cst_1, %cst_2) {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CPU-NEXT:   %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CPU-NEXT:   %2 = stablehlo.select %c, %0, %1 : tensor<4x4xi1>, tensor<4x4xf32>
// CPU-NEXT:   %3 = stablehlo.multiply %cst, %arg1 : tensor<4x4xf32>
// CPU-NEXT:   %4 = stablehlo.add %2, %3 : tensor<4x4xf32>
// CPU-NEXT:   return %4 : tensor<4x4xf32>
// CPU-NEXT: }

// CUDA: func.func @main4(%arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CUDA-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[false, true, true, true], [false, false, true, true], [false, false, false, true], [false, false, false, false]]> : tensor<4x4xi1>
// CUDA-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x4xf32>
// CUDA-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
// CUDA-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CUDA-NEXT:     %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CUDA-NEXT:     %0 = stablehlo.custom_call @cusolver_syrk_ffi(%arg0, %cst_0, %cst_1, %cst_2) {api_version = 4 : i32, backend_config = {transpose = false}, enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>], operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
// CUDA-NEXT:     %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CUDA-NEXT:     %2 = stablehlo.select %c, %0, %1 : tensor<4x4xi1>, tensor<4x4xf32>
// CUDA-NEXT:     %3 = stablehlo.multiply %cst, %arg1 : tensor<4x4xf32>
// CUDA-NEXT:     %4 = stablehlo.add %2, %3 : tensor<4x4xf32>
// CUDA-NEXT:     return %4 : tensor<4x4xf32>
// CUDA-NEXT: }

// TPU: func.func @main4(%arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// TPU-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x4xf32>
// TPU-NEXT:     %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0] : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<4x4xf32>
// TPU-NEXT:     %1 = stablehlo.multiply %cst, %arg1 : tensor<4x4xf32>
// TPU-NEXT:     %2 = stablehlo.add %0, %1 : tensor<4x4xf32>
// TPU-NEXT:     return %2 : tensor<4x4xf32>
// TPU-NEXT: }

module {
    func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<64x64xf32>, %alpha: tensor<f32>, %beta: tensor<f32>) -> tensor<64x64xf32> {
        %0 = enzymexla.blas.syrk %arg0, %arg1, %alpha, %beta {fill, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>} : (tensor<64x32xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
        return %0 : tensor<64x64xf32>
    }
}
