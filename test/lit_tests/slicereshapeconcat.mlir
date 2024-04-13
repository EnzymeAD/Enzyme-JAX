// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_reshape_concat" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
func.func @f1(%314 : tensor<1x1x2048x4x256xbf16>, %344 : tensor<1x1x2048x4x256xbf16>, %371 : tensor<1x1x2048x4x256xbf16>, %392 : tensor<1x1x2048x4x256xbf16>) -> tensor<1x3072x4x256xbf16> {
  %930 = stablehlo.concatenate %314, %344, %371, %392, dim = 2 : (tensor<1x1x2048x4x256xbf16>, tensor<1x1x2048x4x256xbf16>, tensor<1x1x2048x4x256xbf16>, tensor<1x1x2048x4x256xbf16>) -> tensor<1x1x8192x4x256xbf16> 
  %931 = stablehlo.reshape %930 : (tensor<1x1x8192x4x256xbf16>) -> tensor<1x8192x4x256xbf16>
  %932 = stablehlo.slice %931 [0:1, 0:3072, 0:4, 0:256] : (tensor<1x8192x4x256xbf16>) -> tensor<1x3072x4x256xbf16>
  return %932 : tensor<1x3072x4x256xbf16>
}

// CHECK:  func.func @f1(%arg0: tensor<1x1x2048x4x256xbf16>, %arg1: tensor<1x1x2048x4x256xbf16>, %arg2: tensor<1x1x2048x4x256xbf16>, %arg3: tensor<1x1x2048x4x256xbf16>) -> tensor<1x3072x4x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:2048, 0:4, 0:256] : (tensor<1x1x2048x4x256xbf16>) -> tensor<1x1x2048x4x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 0:1, 0:1024, 0:4, 0:256] : (tensor<1x1x2048x4x256xbf16>) -> tensor<1x1x1024x4x256xbf16>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<1x1x2048x4x256xbf16>, tensor<1x1x1024x4x256xbf16>) -> tensor<1x1x3072x4x256xbf16>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x3072x4x256xbf16>) -> tensor<1x3072x4x256xbf16>
// CHECK-NEXT:    return %3 : tensor<1x3072x4x256xbf16>
// CHECK-NEXT:  }
