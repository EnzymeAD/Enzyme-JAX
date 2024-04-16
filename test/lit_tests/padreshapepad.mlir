// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=pad_reshape_pad<1>;pad_pad<1>" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @pad_pad(%30: tensor<1x3072xbf16>, %arg1: tensor<bf16>) -> tensor<1x1x16384x1x1xbf16> {
  %118 = stablehlo.pad %30, %arg1, low = [0, 0], high = [0, 5120], interior = [0, 0] : (tensor<1x3072xbf16>, tensor<bf16>) -> tensor<1x8192xbf16>
  %119 = stablehlo.reshape %118 : (tensor<1x8192xbf16>) -> tensor<1x1x8192x1x1xbf16> 
  %120 = stablehlo.pad %119, %arg1, low = [0, 0, 8192, 0, 0], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x8192x1x1xbf16>, tensor<bf16>) -> tensor<1x1x16384x1x1xbf16> 
  return %120 : tensor<1x1x16384x1x1xbf16> 
}

//CHECK:  func.func @pad_pad(%arg0: tensor<1x3072xbf16>, %arg1: tensor<bf16>) -> tensor<1x1x16384x1x1xbf16> {
//CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1x3072xbf16>) -> tensor<1x1x3072x1x1xbf16>
//CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0, 0, 8192, 0, 0], high = [0, 0, 5120, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x3072x1x1xbf16>, tensor<bf16>) -> tensor<1x1x16384x1x1xbf16>
//CHECK-NEXT:    return %1 : tensor<1x1x16384x1x1xbf16>
//CHECK-NEXT:  }
