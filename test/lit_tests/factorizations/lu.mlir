// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-factorization{backend=cpu})" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-factorization{backend=cuda})" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-factorization{backend=tpu})" %s | FileCheck %s --check-prefix=TPU

// TODO: add for other element types
module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>) {
    %0:3 = enzymexla.lu_factorization %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xi32>, tensor<i32>
  }
}

// TPU: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>) {
// TPU-NEXT:     %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<64xi32>
// TPU-NEXT:     %0:3 = stablehlo.custom_call @LUFactorization(%arg0) : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>)
// TPU-NEXT:     %1 = stablehlo.add %c_0, %0#1 : tensor<64xi32>
// TPU-NEXT:     %2 = stablehlo.is_finite %0#0 : (tensor<64x64xf32>) -> tensor<64x64xi1>
// TPU-NEXT:     %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.and across dimensions = [0, 1] : (tensor<64x64xi1>, tensor<i1>) -> tensor<i1>
// TPU-NEXT:     %4 = stablehlo.not %3 : tensor<i1>
// TPU-NEXT:     %5 = stablehlo.convert %4 : (tensor<i1>) -> tensor<i32>
// TPU-NEXT:     return %0#0, %1, %5 : tensor<64x64xf32>, tensor<64xi32>, tensor<i32>
// TPU-NEXT: }
