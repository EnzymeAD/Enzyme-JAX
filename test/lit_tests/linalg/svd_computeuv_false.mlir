// TODO: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA

module {
  func.func @main1(%arg0: tensor<64x64xf32>) -> (tensor<64xf32>) {
    %0:4 = enzymexla.linalg.svd %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
    return %0#1 : tensor<64xf32>
  }

  func.func @main2(%arg0: tensor<64x64xf32>) -> (tensor<64xf32>) {
    %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<QRIteration>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
    return %0#1 : tensor<64xf32>
  }
}

// CUDA:  module {
// CUDA-NEXT:    func.func @main1(%arg0: tensor<64x64xf32>) -> tensor<64xf32> {
// CUDA-NEXT:      %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {api_version = 4 : i32, backend_config = {compute_uv = false, full_matrices = false}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<i32>)
// CUDA-NEXT:      return %0#1 : tensor<64xf32>
// CUDA-NEXT:    }
// CUDA-NEXT:    func.func @main2(%arg0: tensor<64x64xf32>) -> tensor<64xf32> {
// CUDA-NEXT:      %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {api_version = 4 : i32, backend_config = {compute_uv = false, full_matrices = false, transposed = false}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<i32>)
// CUDA-NEXT:      return %0#1 : tensor<64xf32>
// CUDA-NEXT:    }
// CUDA-NEXT:  }
