// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_all_users_slice},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

module {
    func.func @main(%arg0: tensor<3x4xf32>) -> tensor<3x3xf32> {
        %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
        %1 = stablehlo.slice %0[0:1, 0:3] : (tensor<4x3xf32>) -> tensor<1x3xf32>
        %2 = stablehlo.slice %0[1:2, 0:3] : (tensor<4x3xf32>) -> tensor<1x3xf32>
        %3 = stablehlo.slice %0[2:3, 0:3] : (tensor<4x3xf32>) -> tensor<1x3xf32>
        %4 = stablehlo.transpose %1, dims = [1, 0] : (tensor<1x3xf32>) -> tensor<3x1xf32>
        %5 = stablehlo.transpose %2, dims = [1, 0] : (tensor<1x3xf32>) -> tensor<3x1xf32>
        %6 = stablehlo.transpose %3, dims = [1, 0] : (tensor<1x3xf32>) -> tensor<3x1xf32>
        %7 = stablehlo.concatenate %4, %5, %6, dim = 1 : (tensor<3x1xf32>, tensor<3x1xf32>, tensor<3x1xf32>) -> tensor<3x3xf32>
        return %7 : tensor<3x3xf32>
    }
}

// CHECK: func.func @main(%arg0: tensor<3x4xf32>) -> tensor<3x3xf32> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 0:3] : (tensor<3x4xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:     return %0 : tensor<3x3xf32>
// CHECK-NEXT: }
