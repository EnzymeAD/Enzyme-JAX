// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>) {
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xf32>, tensor<i64>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgeqrf_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(101 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %2 = llvm.call @enzymexla_lapacke_sgeqrf_(%0, %1, %1, %arg0, %1, %arg1) : (i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:    llvm.store %2, %arg2 : i64, !llvm.ptr
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgeqrf_(i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
// CPU-NEXT:    %0:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgeqrf_[[WRAPPER_ID]] (%arg0, %cst, %c) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>)
// CPU-NEXT:    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xf32>, tensor<i64>
// CPU-NEXT:  }

// CUDA: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>) {
// CUDA-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CUDA-NEXT:   %0:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%arg0) {api_version = 4 : i32, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>)
// CUDA-NEXT:   return %0#0, %0#1, %c : tensor<64x64xf32>, tensor<64xf32>, tensor<i64>
// CUDA-NEXT: }

// TPU: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>) {
// TPU-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// TPU-NEXT:   %0:2 = stablehlo.custom_call @Qr(%arg0) : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>)
// TPU-NEXT:   return %0#0, %0#1, %c : tensor<64x64xf32>, tensor<64xf32>, tensor<i64>
// TPU-NEXT: }

module {
  // CPU: func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrf_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf64>, tensor<64xf64>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cgeqrf_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zgeqrf_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<i64>
  }
}
