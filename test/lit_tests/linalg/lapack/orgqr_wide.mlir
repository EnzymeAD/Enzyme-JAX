// RUN: not enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s 2>&1 | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

// LAPACK's ?orgqr/?ungqr need m >= n, so a wide input is rejected on CPU
// CPU: error: 'enzymexla.lapack.orgqr' op `enzymexla.lapack.orgqr` requires at least as many rows as columns

module {
  func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<32xf32>) -> tensor<32x64xf32> {
      %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xf32>, tensor<32xf32>) -> tensor<32x64xf32>
      return %0 : tensor<32x64xf32>
  }
}


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
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xf64>, tensor<32xf64>) -> tensor<32x64xf64>
    return %0 : tensor<32x64xf64>
  }
}

module {
  func.func @main(%arg0: tensor<32x64xcomplex<f32>>, %arg1: tensor<32xcomplex<f32>>) -> tensor<32x64xcomplex<f32>> {
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xcomplex<f32>>, tensor<32xcomplex<f32>>) -> tensor<32x64xcomplex<f32>>
    return %0 : tensor<32x64xcomplex<f32>>
  }
}

module {
  func.func @main(%arg0: tensor<32x64xcomplex<f64>>, %arg1: tensor<32xcomplex<f64>>) -> tensor<32x64xcomplex<f64>> {
    %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<32x64xcomplex<f64>>, tensor<32xcomplex<f64>>) -> tensor<32x64xcomplex<f64>>
    return %0 : tensor<32x64xcomplex<f64>>
  }
}
