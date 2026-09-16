// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<64x128xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x128xf32>) {
    %0:4 = enzymexla.lapack.gesvd %arg0 : (tensor<64x128xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x128xf32>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x128xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<64x128xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
// CHECK-NEXT:   %1:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%0) {api_version = 4 : i32, backend_config = {compute_uv = true, full_matrices = false, transposed = false}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<128x64xf32>) -> (tensor<128x64xf32>, tensor<64xf32>, tensor<128x64xf32>, tensor<64x64xf32>, tensor<i32>)
// CHECK-NEXT:   %2 = stablehlo.transpose %1#2, dims = [1, 0] : (tensor<128x64xf32>) -> tensor<64x128xf32>
// CHECK-NEXT:   return %1#3, %1#1, %2 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x128xf32>
// CHECK-NEXT: }
