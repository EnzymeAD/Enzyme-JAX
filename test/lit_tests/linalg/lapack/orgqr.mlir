// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x64xf32> {
      %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
      return %0 : tensor<64x64xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sorgqr_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:    %1 = llvm.load %arg1 : !llvm.ptr -> i64
// CPU-NEXT:    %2 = llvm.load %arg2 : !llvm.ptr -> i64
// CPU-NEXT:    %3 = llvm.load %arg3 : !llvm.ptr -> i64
// CPU-NEXT:    %4 = llvm.load %arg5 : !llvm.ptr -> i64
// CPU-NEXT:    %5 = llvm.call @enzymexla_lapacke_sorgqr_(%0, %1, %2, %3, %arg4, %4, %arg6) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sorgqr_(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x64xf32> {
// CPU-DAG:    %[[I64_102:[a-z0-9_]+]] = stablehlo.constant dense<102> : tensor<i64>
// CPU-DAG:    %[[I64_64:[a-z0-9_]+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU:        %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sorgqr (%[[I64_102]], %[[I64_64]], %[[I64_64]], %[[I64_64]], %arg0, %[[I64_64]], %arg1) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 4, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<64xf32>) -> tensor<64x64xf32>
// CPU-NEXT:    return %0 : tensor<64x64xf32>
// CPU-NEXT:  }

// CUDA:  func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x64xf32> {
// CUDA-NEXT:    %0 = stablehlo.custom_call @cusolver_orgqr_ffi(%arg0, %arg1) {api_version = 4 : i32, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
// CUDA-NEXT:    return %0 : tensor<64x64xf32>
// CUDA-NEXT:  }

// TPU:  func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x64xf32> {
// TPU-NEXT:    %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
// TPU-NEXT:    return %0 : tensor<64x64xf32>
// TPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x64xf64>, %arg1: tensor<64xf64>) -> tensor<64x64xf64> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dorgqr (
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<64x64xf64>, tensor<64xf64>) -> tensor<64x64xf64>
    return %0 : tensor<64x64xf64>
  }
}

module {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>, %arg1: tensor<64xcomplex<f32>>) -> tensor<64x64xcomplex<f32>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cungqr_ (
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>) -> tensor<64x64xcomplex<f32>>
    return %0 : tensor<64x64xcomplex<f32>>
  }
}

module {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>, %arg1: tensor<64xcomplex<f64>>) -> tensor<64x64xcomplex<f64>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zungqr_ (
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>) -> tensor<64x64xcomplex<f64>>
    return %0 : tensor<64x64xcomplex<f64>>
  }
}
