// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=recognize_insertdim},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=RECOGNIZE
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=lower_insertdim},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_insertdim;insertdim_reshape},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=RESHAPEINSERTDIM

func.func @main(%arg0: tensor<3x2xf32>) -> tensor<3x1x2xf32> {
    // RECOGNIZE:   %0 = enzymexla.insertdim %arg0, dim = 1 : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
    %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
}

func.func @main2(%arg0: tensor<3x2xf32>) -> tensor<3x2x1xf32> {
    // LOWER:   %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<3x2x1xf32>
    %0 = enzymexla.insertdim %arg0, dim = 2 : (tensor<3x2xf32>) -> tensor<3x2x1xf32>
    return %0 : tensor<3x2x1xf32>
}

func.func @main3(%arg0: tensor<3x2xf32>) -> tensor<3x1x2x1xf32> {
    // RESHAPEINSERTDIM:   %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<3x1x2x1xf32>
    %0 = stablehlo.reshape %arg0 : (tensor<3x2xf32>) -> tensor<3x2x1xf32>
    // RESHAPEINSERTDIM-NOT:    enzymexla.insertdim
    %1 = enzymexla.insertdim %0, dim = 1 : (tensor<3x2x1xf32>) -> tensor<3x1x2x1xf32>
    return %1 : tensor<3x1x2x1xf32>
}
