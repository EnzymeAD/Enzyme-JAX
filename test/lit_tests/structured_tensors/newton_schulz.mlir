// RUN: enzymexlamlir-opt --structured-matrix-simplify %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5x4xf32>) -> tensor<5x4xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<2.031500e+00> : tensor<4x4xf32>
    %cst_2 = stablehlo.constant dense<-4.775000e+00> : tensor<4x4xf32>
    %cst_3 = stablehlo.constant dense<3.444500e+00> : tensor<5x4xf32>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_4 = %arg0) : tensor<i64>, tensor<5x4xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %2 = stablehlo.dot_general %iterArg_4, %iterArg_4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<4x4xf32>
      %3 = stablehlo.multiply %cst_2, %2 : tensor<4x4xf32>
      %4 = stablehlo.multiply %cst, %2 : tensor<4x4xf32>
      %5 = stablehlo.dot_general %4, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %6 = stablehlo.add %3, %5 : tensor<4x4xf32>
      %7 = stablehlo.multiply %cst_3, %iterArg_4 : tensor<5x4xf32>
      %8 = stablehlo.dot_general %iterArg_4, %6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x4xf32>, tensor<4x4xf32>) -> tensor<5x4xf32>
      %9 = stablehlo.add %7, %8 : tensor<5x4xf32>
      stablehlo.return %1, %9 : tensor<i64>, tensor<5x4xf32>
    }
    return %0#1 : tensor<5x4xf32>
  }
}
