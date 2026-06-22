// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda blas_int_width=64},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu blas_int_width=64},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=TPU

func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
  %0:4 = enzymexla.lapack.gesvj %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
}

// CUDA: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CUDA-NEXT:     %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {api_version = 4 : i32, backend_config = {compute_uv = true, full_matrices = false}, operand_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi32>)
// CUDA-NEXT:     %1 = stablehlo.convert %0#4 : (tensor<4x3xi32>) -> tensor<4x3xi64>
// CUDA-NEXT:     %2 = stablehlo.transpose %0#3, dims = [0, 1, 3, 2] : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xf32>
// CUDA-NEXT:     return %0#2, %0#1, %2, %1 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CUDA-NEXT:   }

// TPU: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// TPU-NEXT:   %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:   %0:3 = stablehlo.custom_call @SVD(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>)
// TPU-NEXT:   %1 = stablehlo.transpose %0#2, dims = [0, 1, 3, 2] : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xf32>
// TPU-NEXT:   %2 = stablehlo.is_finite %0#0 : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xi1>
// TPU-NEXT:   %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.and across dimensions = [2, 3] : (tensor<4x3x64x64xi1>, tensor<i1>) -> tensor<4x3xi1>
// TPU-NEXT:   %4 = stablehlo.not %3 : tensor<4x3xi1>
// TPU-NEXT:   %5 = stablehlo.convert %4 : (tensor<4x3xi1>) -> tensor<4x3xi64>
// TPU-NEXT:   return %0#0, %0#1, %1, %5 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// TPU-NEXT: }
