// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @one(%330: tensor<1x1x1x1x3072xbf16>, %168 : tensor<bf16>) -> tensor<1x1x1x2048x16384xbf16> {
  %331 = stablehlo.pad %330, %168, low = [0, 0, 0, 0, 8192], high = [0, 0, 0, 0, 5120], interior = [0, 0, 0, 0, 0] : (tensor<1x1x1x1x3072xbf16>, tensor<bf16>) -> tensor<1x1x1x1x16384xbf16>
  %410 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x16384xbf16>) -> tensor<1x1x1x2048x16384xbf16>
  return %410 : tensor<1x1x1x2048x16384xbf16>
}

// CHECK: func.func @one(%arg0: tensor<1x1x1x1x3072xbf16>, %arg1: tensor<bf16>) -> tensor<1x1x1x2048x16384xbf16> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x3072xbf16>) -> tensor<1x1x1x2048x3072xbf16>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0, 0, 0, 0, 8192], high = [0, 0, 0, 0, 5120], interior = [0, 0, 0, 0, 0] : (tensor<1x1x1x2048x3072xbf16>, tensor<bf16>) -> tensor<1x1x1x2048x16384xbf16>
// CHECK-NEXT:    return %1 : tensor<1x1x1x2048x16384xbf16>
// CHECK-NEXT:  }
