// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=CUDA

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
    %0:4 = enzymexla.lapack.gesvj %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
  }
}

// CUDA: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CUDA-NEXT:   %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {api_version = 4 : i32, backend_config = {compute_uv = true, full_matrices = false}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<i32>)
// CUDA-NEXT:   %1 = stablehlo.convert %0#4 : (tensor<i32>) -> tensor<i64>
// CUDA-NEXT:   %2 = stablehlo.transpose %0#3, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CUDA-NEXT:   return %0#2, %0#1, %2, %1 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
// CUDA-NEXT: }
