// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize | FileCheck %s

module {
  func.func @main(%arg9: memref<20x1536x3072xf32, 1>,
                  %arg0: memref<20x1536x3072xf32>) {
    affine.parallel (%arg11, %arg12, %arg13, %arg14) = (0, 0, 0, 0)
        to (4, 16, 95, 3056) {
      %51 = affine.load %arg9[%arg11 + 9, %arg12 + %arg13 * 16 + 9,
                             %arg14 + 8]
          : memref<20x1536x3072xf32, 1>
      %52 = arith.mulf %51, %51 : f32
      affine.store %52, %arg0[%arg11, %arg12 + %arg13 * 16 + 9,
                              %arg14 + 8]
          : memref<20x1536x3072xf32>
    }
    return
  }
}

// CHECK:       func.func private @main_raised(%arg0: tensor<20x1536x3072xf32>, %arg1: tensor<20x1536x3072xf32>) -> (tensor<20x1536x3072xf32>, tensor<20x1536x3072xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [9:13, 9:1529, 8:3064] : (tensor<20x1536x3072xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<4x1520x3056xf32>) -> tensor<4x95x16x3056xf32>
// CHECK-NEXT:    %2 = stablehlo.reshape %0 : (tensor<4x1520x3056xf32>) -> tensor<4x95x16x3056xf32>
// CHECK-NEXT:    %3 = arith.mulf %1, %2 : tensor<4x95x16x3056xf32>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<4x95x16x3056xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg1, %4, %c_1, %c_0, %c : (tensor<20x1536x3072xf32>, tensor<4x1520x3056xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<20x1536x3072xf32>
// CHECK-NEXT:    return %arg0, %5 : tensor<20x1536x3072xf32>, tensor<20x1536x3072xf32>
// CHECK-NEXT:  }
