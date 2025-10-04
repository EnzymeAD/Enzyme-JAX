// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=pad_dot_general<1>(1)},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @fn(%arg0: tensor<f32>) -> (tensor<32x32xf32>, tensor<f32>) {
    %c = stablehlo.constant dense<3> : tensor<32x32xi64>
    %c_0 = stablehlo.constant dense<2> : tensor<32x32xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.pad %0, %cst, low = [2, 3], high = [29, 28], interior = [0, 0] : (tensor<1x1xf32>, tensor<f32>) -> tensor<32x32xf32>
    %2 = stablehlo.convert %c_0 : (tensor<32x32xi64>) -> tensor<32x32xf32>
    %3 = stablehlo.multiply %2, %1 : tensor<32x32xf32>
    %4 = stablehlo.convert %c : (tensor<32x32xi64>) -> tensor<32x32xf32>
    %5 = stablehlo.subtract %3, %4 : tensor<32x32xf32>
    %6 = stablehlo.pad %0, %cst, low = [3, 2], high = [28, 29], interior = [0, 0] : (tensor<1x1xf32>, tensor<f32>) -> tensor<32x32xf32>
    %7 = stablehlo.dot_general %6, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: %7 = stablehlo.dot_general %6, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %7, %arg0 : tensor<32x32xf32>, tensor<f32>
}
