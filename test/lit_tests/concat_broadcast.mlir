// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
module {
  func.func @main(%arg6: tensor<1536xf64>) -> (tensor<1519x3056xf64>, tensor<1x1519x3056xf64>) {
    %633 = stablehlo.slice %arg6 [9:1529] : (tensor<1536xf64>) -> tensor<1520xf64>
    %3184 = stablehlo.slice %arg6 [9:10] : (tensor<1536xf64>) -> tensor<1xf64>

    %3193 = stablehlo.broadcast_in_dim %3184, dims = [0] : (tensor<1xf64>) -> tensor<3056xf64>
    %3194 = stablehlo.reshape %3193 : (tensor<3056xf64>) -> tensor<1x1x3056xf64>
    %3195 = stablehlo.reshape %3193 : (tensor<3056xf64>) -> tensor<1x3056xf64>

    %3196 = stablehlo.broadcast_in_dim %633, dims = [0] : (tensor<1520xf64>) -> tensor<1520x3056xf64>
    %3197 = stablehlo.reshape %3196 : (tensor<1520x3056xf64>) -> tensor<1x1520x3056xf64>
    %3198 = stablehlo.slice %3196 [1:1519, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %3199 = stablehlo.concatenate %3195, %3198, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>) -> tensor<1519x3056xf64>

    %3945 = stablehlo.slice %3197 [0:1, 1:1519, 0:3056] : (tensor<1x1520x3056xf64>) -> tensor<1x1518x3056xf64>
    %3947 = stablehlo.concatenate %3194, %3945, dim = 1 : (tensor<1x1x3056xf64>, tensor<1x1518x3056xf64>) -> tensor<1x1519x3056xf64>

    return %3199, %3947 : tensor<1519x3056xf64>, tensor<1x1519x3056xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1536xf64>) -> (tensor<1519x3056xf64>, tensor<1x1519x3056xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [9:1528] : (tensor<1536xf64>) -> tensor<1519xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1519xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<1519xf64>) -> tensor<1x1519x3056xf64>
// CHECK-NEXT:    return %1, %2 : tensor<1519x3056xf64>, tensor<1x1519x3056xf64>
// CHECK-NEXT:  }
