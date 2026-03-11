// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main(%a: tensor<758xf32>, %b: tensor<4x758x1528xf32>) -> tensor<4x758x1528xf32> {
    %x = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<758xf32>) -> tensor<4x758x1528xf32>
    %y = stablehlo.multiply %x, %b : tensor<4x758x1528xf32>
    return %y : tensor<4x758x1528xf32>
}

// CHECK: func.func @main(%arg0: tensor<758xf32>, %arg1: tensor<4x758x1528xf32>) -> tensor<4x758x1528xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<758xf32>, tensor<4x758x1528xf32>) -> tensor<4x758x1528xf32>
// CHECK-NEXT:     return %0 : tensor<4x758x1528xf32>
// CHECK-NEXT: }
