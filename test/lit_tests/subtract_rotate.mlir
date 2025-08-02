// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

func.func @main2(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.subtract %0, %arg0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

func.func @main3(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %cst_2 = stablehlo.constant dense<-2.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %0, %cst_2 : tensor<1520x3056xf64>
    %2 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %3 = stablehlo.subtract %2, %1 : tensor<1520x3056xf64>
    return %3 : tensor<1520x3056xf64>
}

func.func @main4(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %cst_2 = stablehlo.constant dense<-2.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : si32, dimension = 1 : si32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %0, %cst_2 : tensor<1520x3056xf64>
    %2 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %3 = stablehlo.subtract %1, %2 : tensor<1520x3056xf64>
    return %3 : tensor<1520x3056xf64>
}
