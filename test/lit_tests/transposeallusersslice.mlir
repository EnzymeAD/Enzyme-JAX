// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_all_users_slice},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_all_users_slice},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=CHECK2

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

module {
    func.func @main(%arg0: tensor<6x4x2xf32>) -> tensor<1x2x3xf32> {
        %0 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<6x4x2xf32>) -> tensor<4x2x6xf32>
        %1 = stablehlo.slice %0[0:3, 0:1, 3:5] : (tensor<4x2x6xf32>) -> tensor<3x1x2xf32>
        %2 = stablehlo.slice %0[1:4, 0:1, 3:5] : (tensor<4x2x6xf32>) -> tensor<3x1x2xf32>
        %3 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<3x1x2xf32>) -> tensor<1x2x3xf32>
        %4 = stablehlo.transpose %2, dims = [1, 2, 0] : (tensor<3x1x2xf32>) -> tensor<1x2x3xf32>
        %5 = stablehlo.add %3, %4 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
        return %5 : tensor<1x2x3xf32>
    }
}

// CHECK2: func.func @main(%arg0: tensor<6x4x2xf32>) -> tensor<1x2x3xf32> {
// CHECK2-NEXT:    %0 = stablehlo.slice %arg0 [3:5, 0:3, 0:1] : (tensor<6x4x2xf32>) -> tensor<2x3x1xf32>
// CHECK2-NEXT:    %1 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<2x3x1xf32>) -> tensor<3x1x2xf32>
// CHECK2-NEXT:    %2 = stablehlo.slice %arg0 [3:5, 1:4, 0:1] : (tensor<6x4x2xf32>) -> tensor<2x3x1xf32>
// CHECK2-NEXT:    %3 = stablehlo.transpose %2, dims = [1, 2, 0] : (tensor<2x3x1xf32>) -> tensor<3x1x2xf32>
// CHECK2-NEXT:    %4 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<3x1x2xf32>) -> tensor<1x2x3xf32>
// CHECK2-NEXT:    %5 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<3x1x2xf32>) -> tensor<1x2x3xf32>
// CHECK2-NEXT:    %6 = stablehlo.add %4, %5 : tensor<1x2x3xf32>
// CHECK2-NEXT:    return %6 : tensor<1x2x3xf32>
// CHECK2-NEXT: }
