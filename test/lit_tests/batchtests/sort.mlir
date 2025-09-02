// RUN: enzymexlamlir-opt --enzyme-batch --inline %s | FileCheck %s

module {
  func.func private @unbatched_sort(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<32xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
  func.func @main(%arg0: tensor<4x32xf32>) -> tensor<4x32xf32> {
    %0 = enzyme.batch @unbatched_sort(%arg0) {batch_shape = array<i64: 4>} : (tensor<4x32xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x32xf32>) -> tensor<4x32xf32> {
// CHECK-NEXT:     %0 = "stablehlo.sort"(%arg0) <{dimension = 1 : i64, is_stable = false}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     }) : (tensor<4x32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT:     return %0 : tensor<4x32xf32>
// CHECK-NEXT: }
