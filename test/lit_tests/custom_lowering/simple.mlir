// RUN: enzymexlamlir-opt --resolve-custom-lowering --allow-unregistered-dialect %s | FileCheck %s

func.func @custom_lowering1(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.sine %arg0 : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

func.func @custom_lowering2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

enzymexla.lowering.register "mydialect.sin_or_cos" @custom_lowering1 ({"op" = "sin"})
enzymexla.lowering.register "mydialect.sin_or_cos" @custom_lowering2 ({"op" = "cos"})

func.func @main(%arg0: tensor<8x8xf32>) -> tensor<4x4xf32> {
    %0 = "mydialect.sin_or_cos"(%arg0) {
        "enzymexla.lowering.config" = { "op" = "sin" }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>

    %1 = stablehlo.slice %0 [0:4, 0:4] : (tensor<8x8xf32>) -> tensor<4x4xf32>

    %2 = "mydialect.sin_or_cos"(%1) {
        "enzymexla.lowering.config" = { "op" = "cos" }
    } : (tensor<4x4xf32>) -> tensor<4x4xf32>

    %3 = stablehlo.add %2, %1 : tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
}
