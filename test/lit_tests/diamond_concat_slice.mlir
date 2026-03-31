// RUN: enzymexlamlir-opt %s --enzyme-hlo-ops=passses=131072 | FileCheck %s

module {
  func.func @main(%arg6: tensor<1536xf64>, %3183: tensor<1519x3056xf64>) -> (tensor<1519x3056xf64>, tensor<1519x3056xf64>) {
    %3187 = stablehlo.slice %3183 [0:1, 0:3056] : (tensor<1519x3056xf64>) -> tensor<1x3056xf64>
    %3186 = stablehlo.slice %arg6 [9:10] : (tensor<1536xf64>) -> tensor<1xf64>
    %3196 = stablehlo.broadcast_in_dim %3186, dims = [0] : (tensor<1xf64>) -> tensor<1x3056xf64>
    %3197 = stablehlo.multiply %3196, %3187 : tensor<1x3056xf64>

    %3192 = stablehlo.slice %arg6 [9:1528] : (tensor<1536xf64>) -> tensor<1519xf64>
    %3193 = stablehlo.broadcast_in_dim %3192, dims = [0] : (tensor<1519xf64>) -> tensor<1519x3056xf64>
    %3195 = stablehlo.multiply %3193, %3183 : tensor<1519x3056xf64>
    %3198 = stablehlo.slice %3195 [1:1519, 0:3056] : (tensor<1519x3056xf64>) -> tensor<1518x3056xf64>
    %3199 = stablehlo.concatenate %3197, %3198, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>) -> tensor<1519x3056xf64>

    return %3199, %3195 : tensor<1519x3056xf64>, tensor<1519x3056xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1536xf64>, %arg1: tensor<1519x3056xf64>) -> (tensor<1519x3056xf64>, tensor<1519x3056xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [9:1528] : (tensor<1536xf64>) -> tensor<1519xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1519xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %arg1 : tensor<1519x3056xf64>
// CHECK-NEXT:    return %2, %2 : tensor<1519x3056xf64>, tensor<1519x3056xf64>
// CHECK-NEXT:  }
