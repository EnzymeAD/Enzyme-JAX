// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN

module {
  func.func @main(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
    %1 = stablehlo.subtract %0, %0 : tensor<4x3xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    return %2 : tensor<3x4xf64>
  }
}

// NONAN:  func.func @main(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
// NONAN-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x4xf64>
// NONAN-NEXT:    return %cst : tensor<3x4xf64>
// NONAN-NEXT:  }

// NAN:  func.func @main(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
// NAN-NEXT:    %0 = stablehlo.subtract %arg0, %arg0 : tensor<3x4xf64>
// NAN-NEXT:    return %0 : tensor<3x4xf64>
// NAN-NEXT:  }
