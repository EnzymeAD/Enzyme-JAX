// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_select},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @transpose_select1(%arg0: tensor<3x4x5xf64>, %arg1: tensor<3x4x5xi1>) -> tensor<5x3x4xf64> {
    %0 = stablehlo.negate %arg0 : (tensor<3x4x5xf64>) -> tensor<3x4x5xf64>
    %1 = stablehlo.select %arg1, %0, %arg0 : (tensor<3x4x5xi1>, tensor<3x4x5xf64>, tensor<3x4x5xf64>) -> tensor<3x4x5xf64>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<3x4x5xf64>) -> tensor<5x3x4xf64>
    return %2 : tensor<5x3x4xf64>
}

// CHECK:  func.func @transpose_select1(%arg0: tensor<3x4x5xf64>, %arg1: tensor<3x4x5xi1>) -> tensor<5x3x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.negate %arg0 : tensor<3x4x5xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %arg1, dims = [2, 0, 1] : (tensor<3x4x5xi1>) -> tensor<5x3x4xi1>
// CHECK-NEXT:    %2 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x4x5xf64>) -> tensor<5x3x4xf64>
// CHECK-NEXT: %3 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<3x4x5xf64>) -> tensor<5x3x4xf64>
// CHECK-NEXT:    %4 = stablehlo.select %1, %2, %3 : tensor<5x3x4xi1>, tensor<5x3x4xf64>
// CHECK-NEXT:    return %4 : tensor<5x3x4xf64>
// CHECK-NEXT:  }

func.func @transpose_select2(%arg0: tensor<3x4x5xf64>, %arg1: tensor<i1>) -> tensor<5x3x4xf64> {
    %0 = stablehlo.negate %arg0 : (tensor<3x4x5xf64>) -> tensor<3x4x5xf64>
    %1 = stablehlo.select %arg1, %0, %arg0 : (tensor<i1>, tensor<3x4x5xf64>, tensor<3x4x5xf64>) -> tensor<3x4x5xf64>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<3x4x5xf64>) -> tensor<5x3x4xf64>
    return %2 : tensor<5x3x4xf64>
}

// CHECK:  func.func @transpose_select2(%arg0: tensor<3x4x5xf64>, %arg1: tensor<i1>) -> tensor<5x3x4xf64> {
// CHECK-NEXT:    %0 = stablehlo.negate %arg0 : tensor<3x4x5xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x4x5xf64>) -> tensor<5x3x4xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<3x4x5xf64>) -> tensor<5x3x4xf64>
// CHECK-NEXT:    %3 = stablehlo.select %arg1, %1, %2 : tensor<i1>, tensor<5x3x4xf64>
// CHECK-NEXT:    return %3 : tensor<5x3x4xf64>
// CHECK-NEXT:  }
