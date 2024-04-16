// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @slice_sub(%x: tensor<1x1x8192x1x256xbf16>, %y: tensor<1x1x8192x1x256xbf16>) -> (tensor<1x1x2072x1x20xbf16>, tensor<1x1x2072x1x40xbf16>) {
  %1251 = stablehlo.subtract %x, %y : (tensor<1x1x8192x1x256xbf16>, tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> 
  %1252 = stablehlo.slice %1251 [0:1, 0:1, 1000:3072, 0:1, 0:20] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x2072x1x20xbf16>
  %1253 = stablehlo.slice %1251 [0:1, 0:1, 1000:3072, 0:1, 40:80] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x2072x1x40xbf16>
  return %1252, %1253 : tensor<1x1x2072x1x20xbf16>, tensor<1x1x2072x1x40xbf16>
}

// CHECK:  func.func @slice_sub(%arg0: tensor<1x1x8192x1x256xbf16>, %arg1: tensor<1x1x8192x1x256xbf16>) -> (tensor<1x1x2072x1x20xbf16>, tensor<1x1x2072x1x40xbf16>) {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.slice %arg0 [0:1, 0:1, 1000:3072, 0:1, 0:80] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x2072x1x80xbf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.slice %arg1 [0:1, 0:1, 1000:3072, 0:1, 0:80] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x2072x1x80xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.subtract %[[i0]], %[[i1]] : tensor<1x1x2072x1x80xbf16>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.slice %[[i2]] [0:1, 0:1, 0:2072, 0:1, 40:80] : (tensor<1x1x2072x1x80xbf16>) -> tensor<1x1x2072x1x40xbf16>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.slice %[[i2]] [0:1, 0:1, 0:2072, 0:1, 0:20] : (tensor<1x1x2072x1x80xbf16>) -> tensor<1x1x2072x1x20xbf16>
// CHECK-NEXT:    return %[[i4]], %[[i3]] : tensor<1x1x2072x1x20xbf16>, tensor<1x1x2072x1x40xbf16>
// CHECK-NEXT:  }
