// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @slice_sub(%x: tensor<1x1x8192x1x256xbf16>, %y: tensor<1x1x8192x1x256xbf16>) -> (tensor<1x1x3072x1x256xbf16>) {
  %1251 = stablehlo.subtract %x, %y : (tensor<1x1x8192x1x256xbf16>, tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> 
  %1252 = stablehlo.slice %1251 [0:1, 0:1, 0:3072, 0:1, 0:256] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16>
  return %1252 : tensor<1x1x3072x1x256xbf16>
}

// CHECK:  func.func @slice_sub(%arg0: tensor<1x1x8192x1x256xbf16>, %arg1: tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16> 
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:3072, 0:1, 0:256] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 0:1, 0:3072, 0:1, 0:256] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    return %2 : tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:  }
