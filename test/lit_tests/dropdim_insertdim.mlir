// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=insertdim_dropdim;dropdim_insertdim},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<3x2xf32>
    // CHECK-NOT: enzymexla.insertdim
    %1 = enzymexla.insertdim %0, dims = [1, 4, 3] : (tensor<3x2xf32>) -> tensor<3x1x2x1x1xf32>
    // CHECK-NOT: enzymexla.dropdim
    %2 = enzymexla.dropdim %1, dims = [1, 3, 4] : (tensor<3x1x2x1x1xf32>) -> tensor<3x2xf32>
    // CHECK: return %0 : tensor<3x2xf32>
    return %2 : tensor<3x2xf32>
}

func.func @main2(%arg0: tensor<3x2x1x1xf32>) -> tensor<3x2x1x1xf32> {
    // CHECK-NOT: enzymexla.dropdim
    %0 = enzymexla.dropdim %arg0, dims = [2, 3] : (tensor<3x2x1x1xf32>) -> tensor<3x2xf32>
    // CHECK-NOT: enzymexla.insertdim
    %1 = enzymexla.insertdim %0, dims = [2, 3] : (tensor<3x2xf32>) -> tensor<3x2x1x1xf32>
    // CHECK: %0 = stablehlo.add %arg0, %arg0 : tensor<3x2x1x1xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<3x2x1x1xf32>
    // CHECK: return %0 : tensor<3x2x1x1xf32>
    return %2 : tensor<3x2x1x1xf32>
}
