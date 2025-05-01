// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-factorization{backend=cpu})" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0:3 = enzymexla.lu_factorization %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
    return %0#0 : tensor<64x64xf32>
  }
}
