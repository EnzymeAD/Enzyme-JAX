// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%a : tensor<4x5xf32>, %b : tensor<2x5xf32>) -> tensor<4x5xf32> {
      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %c0 = stablehlo.constant dense<0> : tensor<i32>
      %r = stablehlo.dynamic_update_slice %a, %b, %c1, %c0 : (tensor<4x5xf32>, tensor<2x5xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
    return %r : tensor<4x5xf32>
  }
  func.func @main2(%1565 : tensor<1x1x8192x16x256xbf16>, %1516 : tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
	%84 = stablehlo.constant dense<0> : tensor<i32>
	%1 = stablehlo.constant dense<2048> : tensor<i32>
	%1611 = stablehlo.dynamic_update_slice %1565, %1516, %84, %84, %1, %84, %84 : (tensor<1x1x8192x16x256xbf16>, tensor<1x1x2048x16x256xbf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x8192x16x256xbf16>
    return %1611 : tensor<1x1x8192x16x256xbf16>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<2x5xf32>) -> tensor<4x5xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:5] : (tensor<4x5xf32>) -> tensor<1x5xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [3:4, 0:5] : (tensor<4x5xf32>) -> tensor<1x5xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg1, %1, dim = 0 : (tensor<1x5xf32>, tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:    return %2 : tensor<4x5xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:2048, 0:16, 0:256] : (tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x2048x16x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1, 0:1, 4096:8192, 0:16, 0:256] : (tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x4096x16x256xbf16>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg1, %1, dim = 2 : (tensor<1x1x2048x16x256xbf16>, tensor<1x1x2048x16x256xbf16>, tensor<1x1x4096x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:    return %2 : tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:  }
