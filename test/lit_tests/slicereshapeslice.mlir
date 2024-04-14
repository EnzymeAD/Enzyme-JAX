// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @reshape_slice(%6: tensor<10x30x2048x4xbf16>) -> (tensor<1x2x1024x1x4xbf16>) {
  %7 = stablehlo.slice %6 [7:8, 1:30:10, 0:2048, 0:4] : (tensor<10x30x2048x4xbf16>) -> (tensor<1x3x2048x4xbf16>)
  %8 = stablehlo.reshape %7 : (tensor<1x3x2048x4xbf16>) -> tensor<1x3x2048x1x4xbf16>
  %9 = stablehlo.slice %8 [0:1, 1:3, 0:1024, 0:1, 0:4] : (tensor<1x3x2048x1x4xbf16>) -> tensor<1x2x1024x1x4xbf16>
  return %9 : tensor<1x2x1024x1x4xbf16>
}

// CHECK:  func.func @reshape_slice(%arg0: tensor<10x30x2048x4xbf16>) -> tensor<1x2x1024x1x4xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [7:8, 11:30:10, 0:1024, 0:4] : (tensor<10x30x2048x4xbf16>) -> tensor<1x2x1024x4xbf16>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x2x1024x4xbf16>) -> tensor<1x2x1024x1x4xbf16>
// CHECK-NEXT:    return %1 : tensor<1x2x1024x1x4xbf16>
// CHECK-NEXT:  }
