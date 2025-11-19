// RUN: enzymexlamlir-opt --structured-matrix-simplify %s | FileCheck %s

module {
  func.func @symmetric(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
    %c1 = stablehlo.constant dense<1.000000e+00> : tensor<2x2xf32>
    %cst = stablehlo.add %c0, %c1 : tensor<2x2xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<2x2xf32>
    %2 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %3 = stablehlo.add %1, %2 : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
  }
}
