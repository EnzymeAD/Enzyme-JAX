// RUN: enzymexlamlir-opt --enzymexla-cudnn-hlo-opt %s | FileCheck %s

func.func @dense1(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> (tensor<4x16x16xbf16>) {
    %1 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
    %2 = stablehlo.add %1, %arg2 : tensor<4x16x16xbf16>
    return %2 : tensor<4x16x16xbf16>
}
