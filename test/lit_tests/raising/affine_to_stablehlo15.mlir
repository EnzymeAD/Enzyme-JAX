// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @main(%arg0: memref<4x10xf32>, %arg1: memref<16x10xf32>) {
    affine.parallel (%i) = (0) to (10) {
      affine.for %j = 0 to 4 {
        %0 = affine.load %arg0[%j, %i] : memref<4x10xf32>
        %1 = arith.mulf %0, %0 : f32
        affine.store %1, %arg1[4 * %j, %i] : memref<16x10xf32>
      }
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<4x10xf32>, %arg1: tensor<16x10xf32>) -> (tensor<4x10xf32>, tensor<16x10xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.dynamic_slice %iterArg_2, %iterArg, %c_1, sizes = [1, 10] : (tensor<4x10xf32>, tensor<i64>, tensor<i64>) -> tensor<1x10xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %1 : tensor<1x10xf32>
// CHECK-NEXT:      %3 = stablehlo.multiply %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_update_slice %iterArg_3, %2, %3, %c_1 : (tensor<16x10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<i64>) -> tensor<16x10xf32>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %iterArg_2, %4 : tensor<i64>, tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1, %0#2 : tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:  }
