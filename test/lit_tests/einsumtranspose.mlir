// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3x5xf32>) -> tensor<5x4x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 0, 1] : (tensor<4x3x5xf32>) -> tensor<5x4x3xf32>
    %2 = stablehlo.einsum %0, %1, config = "ba,dbc->cad" : (tensor<3x2xf32>, tensor<5x4x3xf32>) -> tensor<4x2x5xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] : (tensor<4x2x5xf32>) -> tensor<5x4x2xf32>
    func.return %3 : tensor<5x4x2xf32>
  }
}

// CHECK:  func.func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3x5xf32>) -> tensor<5x4x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.einsum %arg0, %arg1, config = "ab,cdb->dca" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<5x4x2xf32>
// CHECK-NEXT:    return %0 : tensor<5x4x2xf32>
// CHECK-NEXT:  }
