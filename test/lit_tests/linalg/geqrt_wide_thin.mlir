// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>) {
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgeqrt_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(102 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(32 : i64) : i64
// CPU-NEXT:    %2 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %3 = llvm.call @enzymexla_lapacke_sgeqrt_(%0, %1, %2, %1, %arg0, %1, %arg1, %1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.store %3, %arg2 : i64, !llvm.ptr
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgeqrt_(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
// CPU-NEXT:    %0:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgeqrt_4 (%arg0, %cst, %c) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>) -> (tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>)
// CPU-NEXT:    return %0#0, %0#1, %0#2 : tensor<32x64xf32>, tensor<32x32xf32>, tensor<i64>
// CPU-NEXT:  }

module {
  // CPU: func.func @main(%arg0: tensor<32x64xf64>) -> (tensor<32x64xf64>, tensor<32x32xf64>, tensor<i64>) {
  func.func @main(%arg0: tensor<32x64xf64>) -> (tensor<32x64xf64>, tensor<32x32xf64>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrt_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<32x64xf64>) -> (tensor<32x64xf64>, tensor<32x32xf64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<32x64xf64>, tensor<32x32xf64>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<32x64xcomplex<f32>>) -> (tensor<32x64xcomplex<f32>>, tensor<32x32xcomplex<f32>>, tensor<i64>) {
  func.func @main(%arg0: tensor<32x64xcomplex<f32>>) -> (tensor<32x64xcomplex<f32>>, tensor<32x32xcomplex<f32>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cgeqrt_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<32x64xcomplex<f32>>) -> (tensor<32x64xcomplex<f32>>, tensor<32x32xcomplex<f32>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<32x64xcomplex<f32>>, tensor<32x32xcomplex<f32>>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<32x64xcomplex<f64>>) -> (tensor<32x64xcomplex<f64>>, tensor<32x32xcomplex<f64>>, tensor<i64>) {
  func.func @main(%arg0: tensor<32x64xcomplex<f64>>) -> (tensor<32x64xcomplex<f64>>, tensor<32x32xcomplex<f64>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zgeqrt_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<32x64xcomplex<f64>>) -> (tensor<32x64xcomplex<f64>>, tensor<32x32xcomplex<f64>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<32x64xcomplex<f64>>, tensor<32x32xcomplex<f64>>, tensor<i64>
  }
}
