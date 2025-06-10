// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// TODO-RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// TODO-RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>) {
    %0:3 = enzymexla.linalg.qr %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>)
    return %0#0, %0#1, %0#3 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>
  }
}

// CPU:  llvm.func @enzymexla_lapacke_dgeqrf_(i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:  func.func @enzymexla_wrapper_lapacke_dgeqrf_[[GEQRF_WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(101 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %2 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %3 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %4 = llvm.call @enzymexla_lapacke_dgeqrf_(%0, %1, %2, %arg0, %3, %arg1) : (i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:    llvm.store %4, %arg2 : i64, !llvm.ptr
// CPU-NEXT:    return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_dorgqr_(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:  func.func @enzymexla_wrapper_lapacke_dorgqr_[[ORGQR_WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(101 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %2 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %3 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %4 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %5 = llvm.call @enzymexla_lapacke_dorgqr_(%0, %1, %2, %3, %arg0, %4, %arg1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:    llvm.store %5, %arg2 : i64, !llvm.ptr
// CPU-NEXT:    return
// CPU-NEXT:  }
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64xf64>
// CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:    %0:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrf_[[GEQRF_WRAPPER_ID]] (%arg0, %cst, %c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]} : (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>)
// CPU-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:    %1:2 = enzymexla.jit_call @enzymexla_wrapper_lapacke_dorgqr_[[ORGQR_WRAPPER_ID]] (%0#0, %0#1, %c_0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 2, operand_tuple_indices = []>]} : (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>) -> (tensor<64x64xf64>, tensor<i64>)
// CPU-NEXT:    return %1#0, %0#0 : tensor<64x64xf64>, tensor<64x64xf64>
// CPU-NEXT:  }

module {
  // CPU: func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrf_[[WRAPPER_ID:[0-9]+]]
    %0:4 = enzymexla.linalg.qr %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#3 : tensor<64x64xf64>, tensor<64xi32>, tensor<i32>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zgeqrf_[[WRAPPER_ID:[0-9]+]]
    %0:4 = enzymexla.linalg.qr %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#3 : tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cgeqrf_[[WRAPPER_ID:[0-9]+]]
    %0:4 = enzymexla.linalg.qr %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#3 : tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>
  }
}
