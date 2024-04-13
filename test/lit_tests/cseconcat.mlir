// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @f1(%314 : tensor<1x1x256xbf16>, %344 : tensor<1x1x256xbf16>) -> (tensor<1x1x512xbf16>, tensor<1x1x512xbf16>) {
  %393 = stablehlo.concatenate %314, %344, dim = 2 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<1x1x512xbf16>
  %930 = stablehlo.concatenate %314, %344, dim = 2 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<1x1x512xbf16>
  return %393, %930 : tensor<1x1x512xbf16>, tensor<1x1x512xbf16>
}

func.func @f2(%314 : tensor<1x1x256xbf16>, %344 : tensor<1x1x256xbf16>) -> (tensor<1x1x512xbf16>, tensor<2x1x256xbf16>) {
  %393 = stablehlo.concatenate %314, %344, dim = 2 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<1x1x512xbf16>
  %930 = stablehlo.concatenate %314, %344, dim = 0 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<2x1x256xbf16>
  return %393, %930 : tensor<1x1x512xbf16>, tensor<2x1x256xbf16>
}

// CHECK:  func.func @f1(%arg0: tensor<1x1x256xbf16>, %arg1: tensor<1x1x256xbf16>) -> (tensor<1x1x512xbf16>, tensor<1x1x512xbf16>) {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<1x1x512xbf16>
// CHECK-NEXT:    return %0, %0 : tensor<1x1x512xbf16>, tensor<1x1x512xbf16>
// CHECK-NEXT:  }
// CHECK:  func.func @f2(%arg0: tensor<1x1x256xbf16>, %arg1: tensor<1x1x256xbf16>) -> (tensor<1x1x512xbf16>, tensor<2x1x256xbf16>) {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<1x1x512xbf16>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1x1x256xbf16>, tensor<1x1x256xbf16>) -> tensor<2x1x256xbf16>
// CHECK-NEXT:    return %0, %1 : tensor<1x1x512xbf16>, tensor<2x1x256xbf16>
// CHECK-NEXT:  }
