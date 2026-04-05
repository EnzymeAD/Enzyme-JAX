// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg6: tensor<1536xf64>, %3931: tensor<1519x3056xf64>) -> (tensor<1519x3056xf64>, tensor<1x1519x3056xf64>) {
    %3184 = stablehlo.slice %arg6 [9:10] : (tensor<1536xf64>) -> tensor<1xf64>
    %3193 = stablehlo.broadcast_in_dim %3184, dims = [0] : (tensor<1xf64>) -> tensor<3056xf64>

    %3194 = stablehlo.reshape %3193 : (tensor<3056xf64>) -> tensor<1x1x3056xf64>
    %3195 = stablehlo.reshape %3193 : (tensor<3056xf64>) -> tensor<1x3056xf64>

    %3189 = stablehlo.slice %arg6 [9:1528] : (tensor<1536xf64>) -> tensor<1519xf64>
    %3190 = stablehlo.broadcast_in_dim %3189, dims = [0] : (tensor<1519xf64>) -> tensor<1519x3056xf64>
    %3191 = stablehlo.reshape %3190 : (tensor<1519x3056xf64>) -> tensor<1x1519x3056xf64>

    %3932 = stablehlo.reshape %3931 : (tensor<1519x3056xf64>) -> tensor<1x1519x3056xf64>

    %3196 = stablehlo.slice %3190 [1:1519, 0:3056] : (tensor<1519x3056xf64>) -> tensor<1518x3056xf64>
    %3197 = stablehlo.concatenate %3195, %3196, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>) -> tensor<1519x3056xf64>

    %3936 = stablehlo.reshape %3931 : (tensor<1519x3056xf64>) -> tensor<1519x1x3056xf64>
    %3937 = stablehlo.slice %3936 [0:1, 0:1, 0:3056] : (tensor<1519x1x3056xf64>) -> tensor<1x1x3056xf64>
    %3943 = stablehlo.multiply %3194, %3937 : tensor<1x1x3056xf64>

    %3942 = stablehlo.multiply %3191, %3932 : tensor<1x1519x3056xf64>
    %3944 = stablehlo.slice %3942 [0:1, 1:1519, 0:3056] : (tensor<1x1519x3056xf64>) -> tensor<1x1518x3056xf64>

    %3945 = stablehlo.concatenate %3943, %3944, dim = 1 : (tensor<1x1x3056xf64>, tensor<1x1518x3056xf64>) -> tensor<1x1519x3056xf64>

    return %3197, %3945 : tensor<1519x3056xf64>, tensor<1x1519x3056xf64>
  }

// CHECK:  func.func @main(%arg0: tensor<1536xf64>, %arg1: tensor<1519x3056xf64>) -> (tensor<1519x3056xf64>, tensor<1x1519x3056xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [9:1528] : (tensor<1536xf64>) -> tensor<1519xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1519xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %arg1 : (tensor<1519x3056xf64>) -> tensor<1x1519x3056xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<1519xf64>) -> tensor<1x1519x3056xf64>
// CHECK-NEXT:    %4 = stablehlo.multiply %3, %2 : tensor<1x1519x3056xf64>
// CHECK-NEXT:    return %1, %4 : tensor<1519x3056xf64>, tensor<1x1519x3056xf64>
// CHECK-NEXT:  }
}
