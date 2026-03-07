// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
      %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xf32>, tensor<32xf32>) -> tensor<32x64xf32>
      return %0 : tensor<32x64xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sorgqr_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(101 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(32 : i64) : i64
// CPU-NEXT:    %2 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %3 = llvm.call @enzymexla_lapacke_sorgqr_(%0, %1, %2, %2, %arg0, %1, %arg1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sorgqr_(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sorgqr_[[WRAPPER_ID]] (%arg0, %arg1) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<32x64xf32>, tensor<32xf32>) -> tensor<32x64xf32>
// CPU-NEXT:    return %0 : tensor<32x64xf32>
// CPU-NEXT:  }

// CUDA:  func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
// CUDA-NEXT:    %0 = stablehlo.custom_call @cusolver_orgqr_ffi(%arg0, %arg1) {api_version = 4 : i32, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<32x64xf32>, tensor<32xf32>) -> tensor<32x64xf32>
// CUDA-NEXT:    return %0 : tensor<32x64xf32>
// CUDA-NEXT:  }

// TPU:  func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
// TPU-NEXT:    %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<32x64xf32>, tensor<32xf32>) -> tensor<32x64xf32>
// TPU-NEXT:    return %0 : tensor<32x64xf32>
// TPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<32x64xf64>, %arg1: tensor<32xf64>) -> tensor<32x64xf64> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dorgqr_[[WRAPPER_ID:[0-9]+]]
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xf64>, tensor<32xf64>) -> tensor<32x64xf64>
    return %0 : tensor<32x64xf64>
  }
}

module {
  func.func @main(%arg0: tensor<32x64xcomplex<f32>>, %arg1: tensor<32xcomplex<f32>>) -> tensor<32x64xcomplex<f32>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cungqr_[[WRAPPER_ID:[0-9]+]]
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xcomplex<f32>>, tensor<32xcomplex<f32>>) -> tensor<32x64xcomplex<f32>>
    return %0 : tensor<32x64xcomplex<f32>>
  }
}

module {
  func.func @main(%arg0: tensor<32x64xcomplex<f64>>, %arg1: tensor<32xcomplex<f64>>) -> tensor<32x64xcomplex<f64>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zungqr_[[WRAPPER_ID:[0-9]+]]
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xcomplex<f64>>, tensor<32xcomplex<f64>>) -> tensor<32x64xcomplex<f64>>
    return %0 : tensor<32x64xcomplex<f64>>
  }
}
