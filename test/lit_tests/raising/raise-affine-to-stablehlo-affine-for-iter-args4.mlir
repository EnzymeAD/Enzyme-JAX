// RUN: enzymexlamlir-opt %s '--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=false},canonicalize,arith-raise,enzyme-hlo-opt)' | FileCheck %s

module {
  func.func @main(%arg0: memref<1x4x10xf32>) {
    %cst = arith.constant 42.0 : f32
    affine.parallel (%i) = (0) to (10) {
      affine.for %j = 0 to 4 {
        %0 = affine.load %arg0[0, %j, %i] : memref<1x4x10xf32>
        %1 = arith.addf %0, %cst : f32
        affine.store %1, %arg0[0, %j, %i] : memref<1x4x10xf32>
      }
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<1x4x10xf32>) -> tensor<1x4x10xf32> {
// CHECK:    %[[CST:.+]] = stablehlo.constant dense<4.200000e+01> : tensor<1x1x10xf32>
// CHECK:    %[[ONE:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:    %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %[[WHILE:.+]]:2 = stablehlo.while(%[[ITER:.+]] = %[[ZERO]], %[[VAL:.+]] = %arg0) : tensor<i64>, tensor<1x4x10xf32>
// CHECK:      %[[SLICE:.+]] = stablehlo.dynamic_slice %[[VAL]], %[[ZERO]], %[[ITER]], %[[ZERO]], sizes = [1, 1, 10] : (tensor<1x4x10xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x10xf32>
// CHECK-NEXT:      %[[ADD:.+]] = stablehlo.add %[[SLICE]], %[[CST]] : tensor<1x1x10xf32>
// CHECK-NEXT:      %[[NEWVAL:.+]] = stablehlo.dynamic_update_slice %[[VAL]], %[[ADD]], %[[ZERO]], %[[ITER]], %[[ZERO]] : (tensor<1x4x10xf32>, tensor<1x1x10xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x4x10xf32>
// CHECK-NEXT:      %[[NEWITER:.+]] = stablehlo.add %[[ITER]], %[[ONE]] {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[NEWITER]], %[[NEWVAL]] : tensor<i64>, tensor<1x4x10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[WHILE]]#1 : tensor<1x4x10xf32>
// CHECK-NEXT:  }
