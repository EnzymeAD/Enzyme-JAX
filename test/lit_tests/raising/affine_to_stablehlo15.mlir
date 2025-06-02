// RUN: enzymexlamlir-opt %s '--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=false},canonicalize,enzyme-hlo-opt{max_constant_expansion=0})' | FileCheck %s

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
// CHECK-NEXT:    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg1) : tensor<i64>, tensor<16x10xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.dynamic_slice %arg0, %iterArg, %c_1, sizes = [1, 10] : (tensor<4x10xf32>, tensor<i64>, tensor<i64>) -> tensor<1x10xf32>
// CHECK-NEXT:      %2 = stablehlo.reshape %1 : (tensor<1x10xf32>) -> tensor<10xf32>
// CHECK-NEXT:      %3 = arith.mulf %2, %2 : tensor<10xf32>
// CHECK-NEXT:      %4 = stablehlo.multiply %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.reshape %3 : (tensor<10xf32>) -> tensor<1x10xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_update_slice %iterArg_2, %5, %4, %c_1 : (tensor<16x10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<i64>) -> tensor<16x10xf32>
// CHECK-NEXT:      %7 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      stablehlo.return %7, %6 : tensor<i64>, tensor<16x10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg0, %0#1 : tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:  }
