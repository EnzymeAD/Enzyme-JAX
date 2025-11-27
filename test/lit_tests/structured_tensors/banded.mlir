// RUN: enzymexlamlir-opt --structured-matrix-simplify %s | FileCheck %s

// func.func @main1(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
//     %c = stablehlo.constant dense<[[true, true, true, true, true, true, true, true, true, true], [true, true, true, true, true, true, true, true, true, true], [true, true, true, true, true, true, true, true, true, true], [false, true, true, true, true, true, true, true, true, true], [false, false, true, true, true, true, true, true, true, true], [false, false, false, true, true, true, true, true, true, true], [false, false, false, false, true, true, true, true, true, true], [false, false, false, false, false, true, true, true, true, true], [false, false, false, false, false, false, true, true, true, true], [false, false, false, false, false, false, false, true, true, true]]> : tensor<10x10xi1>
//     %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
//     %c_0 = stablehlo.constant dense<[[true, true, true, true, false, false, false, false, false, false], [true, true, true, true, true, false, false, false, false, false], [true, true, true, true, true, true, false, false, false, false], [true, true, true, true, true, true, true, false, false, false], [true, true, true, true, true, true, true, true, false, false], [true, true, true, true, true, true, true, true, true, false], [true, true, true, true, true, true, true, true, true, true], [true, true, true, true, true, true, true, true, true, true], [true, true, true, true, true, true, true, true, true, true], [true, true, true, true, true, true, true, true, true, true]]> : tensor<10x10xi1>
//     %0 = stablehlo.select %c_0, %arg0, %cst : tensor<10x10xi1>, tensor<10x10xf32>
//     %1 = stablehlo.select %c, %0, %cst : tensor<10x10xi1>, tensor<10x10xf32>
//     return %1 : tensor<10x10xf32> : tensor<10x10xi1>, tensor<10x10xf32>
// }

func.func @main2(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %c = stablehlo.constant dense<2> : tensor<10x10xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
    %c_0 = stablehlo.constant dense<-3> : tensor<10x10xi64>
    %0 = stablehlo.iota dim = 1 : tensor<10x10xi64>
    %1 = stablehlo.iota dim = 0 : tensor<10x10xi64>
    %2 = stablehlo.subtract %1, %c_0 : tensor<10x10xi64>
    %3 = stablehlo.compare  LE, %0, %2 : (tensor<10x10xi64>, tensor<10x10xi64>) -> tensor<10x10xi1>
    %4 = stablehlo.subtract %1, %c : tensor<10x10xi64>
    %5 = stablehlo.compare  GE, %0, %4 : (tensor<10x10xi64>, tensor<10x10xi64>) -> tensor<10x10xi1>
    %6 = stablehlo.select %3, %arg0, %cst : tensor<10x10xi1>, tensor<10x10xf32>
    %7 = stablehlo.select %5, %6, %cst : tensor<10x10xi1>, tensor<10x10xf32>
    return %7 : tensor<10x10xf32>
}
