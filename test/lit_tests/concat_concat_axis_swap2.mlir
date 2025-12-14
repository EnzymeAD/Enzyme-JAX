// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0 : tensor<64x32xf64>, %arg1 : tensor<1x1xf64>, %arg2 : tensor<1x1xf64>, %arg3 : tensor<1x1xf64>, %arg4 : tensor<1x1xf64>) -> tensor<66x32xf64> {
    %left = stablehlo.slice %arg0 [0:1, 1:31] : (tensor<64x32xf64>) -> tensor<1x30xf64>
    %right = stablehlo.slice %arg0 [63:64, 1:31] : (tensor<64x32xf64>) -> tensor<1x30xf64>
    %mid = stablehlo.slice %arg0 [0:64, 1:31] : (tensor<64x32xf64>) -> tensor<64x30xf64>
    %concat1 = stablehlo.concatenate %left, %mid, %right, dim = 0 : (tensor<1x30xf64>, tensor<64x30xf64>, tensor<1x30xf64>) -> tensor<66x30xf64>

    %top = stablehlo.slice %arg0 [0:64, 0:1] : (tensor<64x32xf64>) -> tensor<64x1xf64>
    %top_full = stablehlo.concatenate %arg1, %top, %arg2, dim = 0 : (tensor<1x1xf64>, tensor<64x1xf64>, tensor<1x1xf64>) -> tensor<66x1xf64>

    %bottom = stablehlo.slice %arg0 [0:64, 31:32] : (tensor<64x32xf64>) -> tensor<64x1xf64>
    %bottom_full = stablehlo.concatenate %arg1, %bottom, %arg2, dim = 0 : (tensor<1x1xf64>, tensor<64x1xf64>, tensor<1x1xf64>) -> tensor<66x1xf64>

    %result = stablehlo.concatenate %top_full, %concat1, %bottom_full, dim = 1 : (tensor<66x1xf64>, tensor<66x30xf64>, tensor<66x1xf64>) -> tensor<66x32xf64>
    return %result : tensor<66x32xf64>
}

// CHECK: func.func @main(%arg0: tensor<64x32xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<1x1xf64>, %arg3: tensor<1x1xf64>, %arg4: tensor<1x1xf64>) -> tensor<66x32xf64> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:1, 1:31] : (tensor<64x32xf64>) -> tensor<1x30xf64>
// CHECK-NEXT:     %1 = stablehlo.slice %arg0 [63:64, 1:31] : (tensor<64x32xf64>) -> tensor<1x30xf64>
// CHECK-NEXT:     %2 = stablehlo.concatenate %arg1, %0, %arg1, dim = 1 : (tensor<1x1xf64>, tensor<1x30xf64>, tensor<1x1xf64>) -> tensor<1x32xf64>
// CHECK-NEXT:     %3 = stablehlo.concatenate %arg2, %1, %arg2, dim = 1 : (tensor<1x1xf64>, tensor<1x30xf64>, tensor<1x1xf64>) -> tensor<1x32xf64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %arg0, %3, dim = 0 : (tensor<1x32xf64>, tensor<64x32xf64>, tensor<1x32xf64>) -> tensor<66x32xf64>
// CHECK-NEXT:     return %4 : tensor<66x32xf64>
// CHECK-NEXT:   }