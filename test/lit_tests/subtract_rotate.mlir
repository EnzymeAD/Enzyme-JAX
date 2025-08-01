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
