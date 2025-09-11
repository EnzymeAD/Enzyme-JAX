// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=slice_to_batch_dotgeneral},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
//     %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
//     %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
//     %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
//     %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
//     %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
//     %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
//     %6 = stablehlo.dot_general %arg0, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
//     %7 = stablehlo.reshape %6 : (tensor<4x1xf32>) -> tensor<4xf32>
//     %8 = stablehlo.dot_general %arg0, %4, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
//     %9 = stablehlo.reshape %8 : (tensor<4x1xf32>) -> tensor<4xf32>
//     %10 = stablehlo.dot_general %arg0, %3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
//     %11 = stablehlo.reshape %10 : (tensor<4x1xf32>) -> tensor<4xf32>
//     %12 = stablehlo.dot_general %arg0, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
//     %13 = stablehlo.reshape %12 : (tensor<4x1xf32>) -> tensor<4xf32>
//     %14 = stablehlo.dot_general %arg0, %1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
//     %15 = stablehlo.reshape %14 : (tensor<4x1xf32>) -> tensor<4xf32>
//     %16 = stablehlo.dot_general %arg0, %0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
//     %17 = stablehlo.reshape %16 : (tensor<4x1xf32>) -> tensor<4xf32>
//     return %7, %9, %11, %13, %15, %17 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// }


func.func @main2(%arg0: tensor<3x4xf32>, %arg1: tensor<6x3xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.slice %arg1 [5:6, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.slice %arg1 [4:5, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %2 = stablehlo.slice %arg1 [3:4, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.slice %arg1 [2:3, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %4 = stablehlo.slice %arg1 [1:2, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %5 = stablehlo.slice %arg1 [0:1, 0:3] : (tensor<6x3xf32>) -> tensor<1x3xf32>
    %6 = stablehlo.dot_general %arg0, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %8 = stablehlo.dot_general %arg0, %4, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %10 = stablehlo.dot_general %arg0, %3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %12 = stablehlo.dot_general %arg0, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %14 = stablehlo.dot_general %arg0, %1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %16 = stablehlo.dot_general %arg0, %0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x3xf32>) -> tensor<4x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<4x1xf32>) -> tensor<4xf32>
    %9 = stablehlo.reshape %8 : (tensor<4x1xf32>) -> tensor<4xf32>
    %11 = stablehlo.reshape %10 : (tensor<4x1xf32>) -> tensor<4xf32>
    %13 = stablehlo.reshape %12 : (tensor<4x1xf32>) -> tensor<4xf32>
    %15 = stablehlo.reshape %14 : (tensor<4x1xf32>) -> tensor<4xf32>
    %17 = stablehlo.reshape %16 : (tensor<4x1xf32>) -> tensor<4xf32>
    return %7, %9, %11, %13, %15, %17 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}
