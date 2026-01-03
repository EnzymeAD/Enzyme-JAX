// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

// Test that a chain of add operations with dynamic slices at constant offsets
// is raised to a reduce_window operation.

module {
  func.func @main(%arg0: tensor<64xf32>) -> tensor<1xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0.333333343> : tensor<1xf32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<62> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.constant dense<0.0> : tensor<1xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %0) : tensor<i64>, tensor<1xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      // Compute indices at offsets -1, 0, +1 from the induction variable
      %idx = stablehlo.add %c_3, %iterArg {enzymexla.bounds = [[1, 62]]} : tensor<i64>
      %idx_i32 = stablehlo.convert %idx {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
      
      // Slice at offset -1 (idx - 1)
      %idx_m1 = stablehlo.subtract %idx_i32, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
      %slice_m1 = stablehlo.dynamic_slice %arg0, %idx_m1, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
      
      // Slice at offset 0 (idx itself as offset)
      %slice_0 = stablehlo.dynamic_slice %arg0, %idx_i32, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
      
      // Slice at offset +1 (idx + 1)
      %idx_p1 = stablehlo.add %idx_i32, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
      %slice_p1 = stablehlo.dynamic_slice %arg0, %idx_p1, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
      
      // Chain of adds
      %add1 = stablehlo.add %slice_m1, %slice_0 : tensor<1xf32>
      %add2 = stablehlo.add %add1, %slice_p1 : tensor<1xf32>
      
      // Scale by constant
      %result = stablehlo.multiply %c_0, %add2 : tensor<1xf32>
      
      %next_idx = stablehlo.add %iterArg, %c_3 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
      stablehlo.return %next_idx, %result : tensor<i64>, tensor<1xf32>
    }
    return %1#1 : tensor<1xf32>
  }
}

// CHECK: %0 = "stablehlo.reduce_window"(%arg0, %cst_3) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:   %4 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %4 : tensor<f32>
// CHECK-NEXT: }) : (tensor<64xf32>, tensor<f32>) -> tensor<62xf32>
// CHECK-NEXT: %1 = stablehlo.reshape %0 : (tensor<62xf32>) -> tensor<62x1xf32>
// CHECK-NEXT: %2 = stablehlo.multiply %cst, %1 : tensor<62x1xf32>
