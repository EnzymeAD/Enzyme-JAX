// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=recognize_dropdim},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=RECOGNIZE
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=lower_dropdim},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_dropdim;dropdim_reshape},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=RESHAPEDROPDIM

func.func @main(%arg0: tensor<3x1x2xf32>) -> tensor<3x2xf32> {
    // RECOGNIZE:   %0 = enzymexla.dropdim %arg0, dims = [1] : (tensor<3x1x2xf32>) -> tensor<3x2xf32>
    %0 = stablehlo.reshape %arg0 : (tensor<3x1x2xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
}

func.func @main2(%arg0: tensor<3x1x2xf32>) -> tensor<3x2xf32> {
    // LOWER:    %0 = stablehlo.reshape %arg0 : (tensor<3x1x2xf32>) -> tensor<3x2xf32>
    %0 = enzymexla.dropdim %arg0, dims = [1] : (tensor<3x1x2xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
}

func.func @main3(%arg0: tensor<3x1x2x1xf32>) -> tensor<3x2xf32> {
    // RESHAPEDROPDIM:   %0 = stablehlo.reshape %arg0 : (tensor<3x1x2x1xf32>) -> tensor<3x2xf32>
    %0 = stablehlo.reshape %arg0 : (tensor<3x1x2x1xf32>) -> tensor<3x2x1xf32>
    // RESHAPEDROPDIM-NOT: enzymexla.dropdim
    %1 = enzymexla.dropdim %0, dims = [2] : (tensor<3x2x1xf32>) -> tensor<3x2xf32>
    return %1 : tensor<3x2xf32>
}
