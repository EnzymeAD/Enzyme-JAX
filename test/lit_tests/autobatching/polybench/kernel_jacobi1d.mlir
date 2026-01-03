// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}) -> (tensor<64xf32>, tensor<64xf32>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.333333343> : tensor<1xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<32> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<62> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_5 = %arg0, %iterArg_6 = %arg1) : tensor<i64>, tensor<64xf32>, tensor<64xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_3 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      // full indexing along that dimension
      %2:2 = stablehlo.while(%iterArg_7 = %c_1, %iterArg_8 = %iterArg_6) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %4 = stablehlo.compare  LT, %iterArg_7, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %4 : tensor<i1>
      } do {
        %4 = stablehlo.add %c_0, %iterArg_7 {enzymexla.bounds = [[2, 63]]} : tensor<i64>
        %5 = stablehlo.add %iterArg_7, %c_3 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %6 = stablehlo.subtract %4, %c_3 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %7 = stablehlo.convert %6 {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
        %9 = stablehlo.dynamic_slice %iterArg_5, %8, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %10 = stablehlo.convert %4 {enzymexla.bounds = [[2, 63]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c {enzymexla.bounds = [[1, 62]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %iterArg_5, %11, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %13 = stablehlo.add %4, %c_3 {enzymexla.bounds = [[3, 64]]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[3, 64]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %iterArg_5, %15, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.add %9, %12 : tensor<1xf32>
        %18 = stablehlo.add %17, %16 : tensor<1xf32>
        %19 = stablehlo.multiply %cst, %18 : tensor<1xf32>
        %20 = stablehlo.dynamic_update_slice %iterArg_8, %19, %11 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %5, %20 : tensor<i64>, tensor<64xf32>
      }
      %3:2 = stablehlo.while(%iterArg_7 = %c_1, %iterArg_8 = %iterArg_5) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %4 = stablehlo.compare  LT, %iterArg_7, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %4 : tensor<i1>
      } do {
        %4 = stablehlo.add %c_0, %iterArg_7 {enzymexla.bounds = [[2, 63]]} : tensor<i64>
        %5 = stablehlo.add %iterArg_7, %c_3 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %6 = stablehlo.subtract %4, %c_3 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %7 = stablehlo.convert %6 {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
        %9 = stablehlo.dynamic_slice %2#1, %8, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %10 = stablehlo.convert %4 {enzymexla.bounds = [[2, 63]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c {enzymexla.bounds = [[1, 62]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %2#1, %11, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %13 = stablehlo.add %4, %c_3 {enzymexla.bounds = [[3, 64]]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[3, 64]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %2#1, %15, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.add %9, %12 : tensor<1xf32>
        %18 = stablehlo.add %17, %16 : tensor<1xf32>
        %19 = stablehlo.multiply %cst, %18 : tensor<1xf32>
        %20 = stablehlo.dynamic_update_slice %iterArg_8, %19, %11 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %5, %20 : tensor<i64>, tensor<64xf32>
      }
      stablehlo.return %1, %3#1, %2#1 : tensor<i64>, tensor<64xf32>, tensor<64xf32>
    }
    return %0#1, %0#2 : tensor<64xf32>, tensor<64xf32>
  }
}

// CHECK: %9 = stablehlo.slice %arg0 [0:1] : (tensor<64xf32>) -> tensor<1xf32>
// CHECK-NEXT: %10 = stablehlo.slice %arg0 [63:64] : (tensor<64xf32>) -> tensor<1xf32>
// CHECK-NEXT: %11 = stablehlo.concatenate %9, %iterArg_3, %10, dim = 0 : (tensor<1xf32>, tensor<62xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK-NEXT: %12 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT: %13 = "stablehlo.reduce_window"(%11, %cst_0) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %20 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %20 : tensor<f32>
// CHECK-NEXT: }) : (tensor<64xf32>, tensor<f32>) -> tensor<62xf32>
// CHECK-NEXT: %14 = stablehlo.multiply %cst, %13 : tensor<62xf32>
// CHECK-NEXT: %15 = stablehlo.slice %arg1 [0:1] : (tensor<64xf32>) -> tensor<1xf32>
// CHECK-NEXT: %16 = stablehlo.slice %arg1 [63:64] : (tensor<64xf32>) -> tensor<1xf32>
// CHECK-NEXT: %17 = stablehlo.concatenate %15, %14, %16, dim = 0 : (tensor<1xf32>, tensor<62xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK-NEXT: %18 = "stablehlo.reduce_window"(%17, %cst_0) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %20 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %20 : tensor<f32>
// CHECK-NEXT: }) : (tensor<64xf32>, tensor<f32>) -> tensor<62xf32>
// CHECK-NEXT: %19 = stablehlo.multiply %cst, %18 : tensor<62xf32>

module {
  func.func @main(%arg0: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}) -> (tensor<64xf32>, tensor<64xf32>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.333333343> : tensor<1xf32>
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<32> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %c_5 = stablehlo.constant dense<56> : tensor<i64>
    %c_6 = stablehlo.constant dense<55> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c_2, %iterArg_7 = %arg0, %iterArg_8 = %arg1) : tensor<i64>, tensor<64xf32>, tensor<64xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_4 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %2:2 = stablehlo.while(%iterArg_9 = %c_2, %iterArg_10 = %iterArg_8) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %4 = stablehlo.compare  LT, %iterArg_9, %c_5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %4 : tensor<i1>
      } do {
        %4 = stablehlo.add %c_1, %iterArg_9 {enzymexla.bounds = [[5, 60]]} : tensor<i64>
        %5 = stablehlo.add %iterArg_9, %c_4 {enzymexla.bounds = [[1, 56]]} : tensor<i64>
        %6 = stablehlo.subtract %4, %c_4 {enzymexla.bounds = [[4, 59]]} : tensor<i64>
        %7 = stablehlo.convert %6 {enzymexla.bounds = [[4, 59]]} : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c_0 {enzymexla.bounds = [[3, 58]]} : tensor<i32>
        %9 = stablehlo.dynamic_slice %iterArg_7, %8, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %10 = stablehlo.convert %4 {enzymexla.bounds = [[5, 60]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c_0 {enzymexla.bounds = [[4, 59]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %iterArg_7, %11, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %13 = stablehlo.add %4, %c_4 {enzymexla.bounds = [[6, 61]]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[6, 61]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c_0 {enzymexla.bounds = [[5, 60]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %iterArg_7, %15, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.add %9, %12 : tensor<1xf32>
        %18 = stablehlo.add %17, %16 : tensor<1xf32>
        %19 = stablehlo.multiply %cst, %18 : tensor<1xf32>
        %20 = stablehlo.dynamic_update_slice %iterArg_10, %19, %11 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %5, %20 : tensor<i64>, tensor<64xf32>
      }
      %3:2 = stablehlo.while(%iterArg_9 = %c_2, %iterArg_10 = %iterArg_7) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %4 = stablehlo.compare  LT, %iterArg_9, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %4 : tensor<i1>
      } do {
        %4 = stablehlo.add %c, %iterArg_9 {enzymexla.bounds = [[3, 57]]} : tensor<i64>
        %5 = stablehlo.add %iterArg_9, %c_4 {enzymexla.bounds = [[1, 55]]} : tensor<i64>
        %6 = stablehlo.subtract %4, %c_4 {enzymexla.bounds = [[2, 56]]} : tensor<i64>
        %7 = stablehlo.convert %6 {enzymexla.bounds = [[2, 56]]} : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c_0 {enzymexla.bounds = [[1, 55]]} : tensor<i32>
        %9 = stablehlo.dynamic_slice %2#1, %8, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %10 = stablehlo.convert %4 {enzymexla.bounds = [[3, 57]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c_0 {enzymexla.bounds = [[2, 56]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %2#1, %11, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %13 = stablehlo.add %4, %c_4 {enzymexla.bounds = [[4, 58]]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[4, 58]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c_0 {enzymexla.bounds = [[3, 57]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %2#1, %15, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.add %9, %12 : tensor<1xf32>
        %18 = stablehlo.add %17, %16 : tensor<1xf32>
        %19 = stablehlo.multiply %cst, %18 : tensor<1xf32>
        %20 = stablehlo.dynamic_update_slice %iterArg_10, %19, %11 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %5, %20 : tensor<i64>, tensor<64xf32>
      }
      stablehlo.return %1, %3#1, %2#1 : tensor<i64>, tensor<64xf32>, tensor<64xf32>
    }
    return %0#1, %0#2 : tensor<64xf32>, tensor<64xf32>
  }
}

// CHECK: %9 = stablehlo.add %iterArg, %c_3 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT: %10 = stablehlo.slice %iterArg_4 [1:55] : (tensor<55xf32>) -> tensor<54xf32>
// CHECK-NEXT: %11 = stablehlo.slice %arg0 [57:61] : (tensor<64xf32>) -> tensor<4xf32>
// CHECK-NEXT: %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<54xf32>, tensor<4xf32>) -> tensor<58xf32>
// CHECK-NEXT: %13 = "stablehlo.reduce_window"(%12, %cst_1) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %20 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %20 : tensor<f32>
// CHECK-NEXT: }) : (tensor<58xf32>, tensor<f32>) -> tensor<56xf32>
// CHECK-NEXT: %14 = stablehlo.multiply %cst_0, %13 : tensor<56xf32>
// CHECK-NEXT: %15 = stablehlo.slice %arg1 [1:4] : (tensor<64xf32>) -> tensor<3xf32>
// CHECK-NEXT: %16 = stablehlo.slice %14 [0:54] : (tensor<56xf32>) -> tensor<54xf32>
// CHECK-NEXT: %17 = stablehlo.concatenate %15, %16, dim = 0 : (tensor<3xf32>, tensor<54xf32>) -> tensor<57xf32>
// CHECK-NEXT: %18 = "stablehlo.reduce_window"(%17, %cst_1) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %20 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %20 : tensor<f32>
// CHECK-NEXT: }) : (tensor<57xf32>, tensor<f32>) -> tensor<55xf32>
// CHECK-NEXT: %19 = stablehlo.multiply %cst, %18 : tensor<55xf32>
// CHECK-NEXT: stablehlo.return %9, %19, %14 : tensor<i64>, tensor<55xf32>, tensor<56xf32>

module {
  func.func @main(%arg0: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}) -> (tensor<64xf32>, tensor<64xf32>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.333333343> : tensor<1xf32>
    %c = stablehlo.constant dense<2> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %c_2 = stablehlo.constant dense<5> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %c_6 = stablehlo.constant dense<56> : tensor<i64>
    %c_7 = stablehlo.constant dense<55> : tensor<i64>
    %c_8 = stablehlo.constant dense<32> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c_4, %iterArg_9 = %arg0, %iterArg_10 = %arg1) : tensor<i64>, tensor<64xf32>, tensor<64xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_5 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %2:2 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %iterArg_10) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %4 = stablehlo.compare  LT, %iterArg_11, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %4 : tensor<i1>
      } do {
        %4 = stablehlo.add %c_2, %iterArg_11 {enzymexla.bounds = [[5, 60]]} : tensor<i64>
        %5 = stablehlo.add %iterArg_11, %c_5 {enzymexla.bounds = [[1, 56]]} : tensor<i64>
        %6 = stablehlo.subtract %4, %c_1 {enzymexla.bounds = [[2, 57]]} : tensor<i64>
        %7 = stablehlo.convert %6 {enzymexla.bounds = [[2, 57]]} : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c_0 {enzymexla.bounds = [[1, 56]]} : tensor<i32>
        %9 = stablehlo.dynamic_slice %iterArg_9, %8, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %10 = stablehlo.convert %4 {enzymexla.bounds = [[5, 60]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c_0 {enzymexla.bounds = [[4, 59]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %iterArg_9, %11, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %13 = stablehlo.add %4, %c_1 {enzymexla.bounds = [[8, 63]]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[8, 63]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c_0 {enzymexla.bounds = [[7, 62]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %iterArg_9, %15, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.add %9, %12 : tensor<1xf32>
        %18 = stablehlo.add %17, %16 : tensor<1xf32>
        %19 = stablehlo.multiply %cst, %18 : tensor<1xf32>
        %20 = stablehlo.dynamic_update_slice %iterArg_12, %19, %11 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %5, %20 : tensor<i64>, tensor<64xf32>
      }
      %3:2 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %iterArg_9) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %4 = stablehlo.compare  LT, %iterArg_11, %c_7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %4 : tensor<i1>
      } do {
        %4 = stablehlo.add %c_1, %iterArg_11 {enzymexla.bounds = [[3, 57]]} : tensor<i64>
        %5 = stablehlo.add %iterArg_11, %c_5 {enzymexla.bounds = [[1, 55]]} : tensor<i64>
        %6 = stablehlo.subtract %4, %c {enzymexla.bounds = [[1, 55]]} : tensor<i64>
        %7 = stablehlo.convert %6 {enzymexla.bounds = [[1, 55]]} : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c_0 {enzymexla.bounds = [[0, 54]]} : tensor<i32>
        %9 = stablehlo.dynamic_slice %2#1, %8, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %10 = stablehlo.convert %4 {enzymexla.bounds = [[3, 57]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c_0 {enzymexla.bounds = [[2, 56]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %2#1, %11, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %13 = stablehlo.add %4, %c {enzymexla.bounds = [[5, 59]]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[5, 59]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c_0 {enzymexla.bounds = [[4, 58]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %2#1, %15, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %17 = stablehlo.add %4, %c_3 {enzymexla.bounds = [[7, 61]]} : tensor<i64>
        %18 = stablehlo.convert %17 {enzymexla.bounds = [[7, 61]]} : (tensor<i64>) -> tensor<i32>
        %19 = stablehlo.subtract %18, %c_0 {enzymexla.bounds = [[6, 60]]} : tensor<i32>
        %20 = stablehlo.dynamic_slice %2#1, %19, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %21 = stablehlo.add %9, %12 : tensor<1xf32>
        %22 = stablehlo.add %21, %16 : tensor<1xf32>
        %23 = stablehlo.add %22, %20 : tensor<1xf32>
        %24 = stablehlo.multiply %cst, %23 : tensor<1xf32>
        %25 = stablehlo.dynamic_update_slice %iterArg_12, %24, %11 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %5, %25 : tensor<i64>, tensor<64xf32>
      }
      stablehlo.return %1, %3#1, %2#1 : tensor<i64>, tensor<64xf32>, tensor<64xf32>
    }
    return %0#1, %0#2 : tensor<64xf32>, tensor<64xf32>
  }
}

// CHECK: %9 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT: %10 = stablehlo.slice %arg0 [1:2] : (tensor<64xf32>) -> tensor<1xf32>
// CHECK-NEXT: %11 = stablehlo.slice %arg0 [57:63] : (tensor<64xf32>) -> tensor<6xf32>
// CHECK-NEXT: %12 = stablehlo.concatenate %10, %iterArg_4, %11, dim = 0 : (tensor<1xf32>, tensor<55xf32>, tensor<6xf32>) -> tensor<62xf32>
// CHECK-NEXT: %13 = "stablehlo.reduce_window"(%12, %cst_1) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 3>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %20 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %20 : tensor<f32>
// CHECK-NEXT: }) : (tensor<62xf32>, tensor<f32>) -> tensor<56xf32>
// CHECK-NEXT: %14 = stablehlo.multiply %cst_0, %13 : tensor<56xf32>
// CHECK-NEXT: %15 = stablehlo.slice %arg1 [0:4] : (tensor<64xf32>) -> tensor<4xf32>
// CHECK-NEXT: %16 = stablehlo.slice %arg1 [60:61] : (tensor<64xf32>) -> tensor<1xf32>
// CHECK-NEXT: %17 = stablehlo.concatenate %15, %14, %16, dim = 0 : (tensor<4xf32>, tensor<56xf32>, tensor<1xf32>) -> tensor<61xf32>
// CHECK-NEXT: %18 = "stablehlo.reduce_window"(%17, %cst_1) <{base_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 2>, window_dimensions = array<i64: 4>, window_strides = array<i64: 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %20 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %20 : tensor<f32>
// CHECK-NEXT: }) : (tensor<61xf32>, tensor<f32>) -> tensor<55xf32>
// CHECK-NEXT: %19 = stablehlo.multiply %cst, %18 : tensor<55xf32>
// CHECK-NEXT: stablehlo.return %9, %19, %14 : tensor<i64>, tensor<55xf32>, tensor<56xf32>
