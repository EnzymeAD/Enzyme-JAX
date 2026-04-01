// RUN: enzymexlamlir-opt %s '--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=false},canonicalize,enzyme-hlo-opt{max_constant_expansion=0})' | FileCheck %s
// RUN: enzymexlamlir-opt %s '--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=false},canonicalize,enzyme-hlo-opt{max_constant_expansion=0 enable_auto_batching_passes=true})' | FileCheck %s --check-prefix=LOOPRAISE

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
// CHECK-NEXT:      %2 = arith.mulf %1, %1 : tensor<1x10xf32>
// CHECK-NEXT:      %3 = stablehlo.multiply %iterArg, %c_0 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.dynamic_update_slice %iterArg_2, %2, %3, %c_1 : (tensor<16x10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<i64>) -> tensor<16x10xf32>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %4 : tensor<i64>, tensor<16x10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg0, %0#1 : tensor<4x10xf32>, tensor<16x10xf32>
// CHECK-NEXT:  }

// LOOPRAISE: func.func private @main_raised(%arg0: tensor<4x10xf32>, %arg1: tensor<16x10xf32>) -> (tensor<4x10xf32>, tensor<16x10xf32>) {
// LOOPRAISE-NEXT:   %c = stablehlo.constant dense<4> : tensor<4x1xi64>
// LOOPRAISE-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// LOOPRAISE-NEXT:   %0 = arith.mulf %arg0, %arg0 : tensor<4x10xf32>
// LOOPRAISE-NEXT:   %1 = stablehlo.iota dim = 0 : tensor<4x1xi64>
// LOOPRAISE-NEXT:   %2 = stablehlo.multiply %1, %c : tensor<4x1xi64>
// LOOPRAISE-NEXT:   %3 = stablehlo.pad %2, %c_0, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x1xi64>, tensor<i64>) -> tensor<4x2xi64>
// LOOPRAISE-NEXT:   %4 = "stablehlo.scatter"(%arg1, %3, %0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// LOOPRAISE-NEXT:   ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// LOOPRAISE-NEXT:     stablehlo.return %arg3 : tensor<f32>
// LOOPRAISE-NEXT:   }) : (tensor<16x10xf32>, tensor<4x2xi64>, tensor<4x10xf32>) -> tensor<16x10xf32>
// LOOPRAISE-NEXT:   return %arg0, %4 : tensor<4x10xf32>, tensor<16x10xf32>
// LOOPRAISE-NEXT: }
