// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN

func.func @pad_div(%4: tensor<1x3x1024xf32>, %2: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
  %constant_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %5 = stablehlo.pad %4, %constant_0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %7 = stablehlo.divide %5, %2 : tensor<1x3x2048xf32>
  return %7 : tensor<1x3x2048xf32>
}

// NONAN:  func.func @pad_div(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
// NONAN-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// NONAN-NEXT:    %[[i1:.+]] = stablehlo.slice %arg1 [0:1, 0:3, 1024:2048] : (tensor<1x3x2048xf32>) -> tensor<1x3x1024xf32>
// NONAN-NEXT:    %[[i2:.+]] = stablehlo.divide %arg0, %[[i1]] : tensor<1x3x1024xf32>
// NONAN-NEXT:    %[[i3:.+]] = stablehlo.pad %[[i2]], %[[i0]], low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
// NONAN-NEXT:    return %[[i3]] : tensor<1x3x2048xf32>
// NONAN-NEXT:  }

// NAN:  func.func @pad_div(%arg0: tensor<1x3x1024xf32>, %arg1: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
// NAN-NOT: stablehlo.slice
// NAN:    stablehlo.pad
// NAN:    stablehlo.divide
