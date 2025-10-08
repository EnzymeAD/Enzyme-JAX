// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32> {enzymexla.memory_effects = []}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzymexla.disable_min_cut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %6 = stablehlo.add %5, %c_2 : tensor<i64>
      %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_3, %4, %8 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %9 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32> {enzymexla.memory_effects = []}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     return %arg1 : tensor<10xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32> {enzymexla.memory_effects = []}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<9> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzymexla.disable_min_cut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %6 = stablehlo.add %5, %c_2 : tensor<i64>
      %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_3, %4, %8 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %9 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32> {enzymexla.memory_effects = []}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:      %0 = stablehlo.slice %arg1 [0:9] : (tensor<10xf32>) -> tensor<9xf32>
// CHECK-NEXT:      %1 = stablehlo.slice %arg0 [9:10] : (tensor<10xf32>) -> tensor<1xf32>
// CHECK-NEXT:      %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<9xf32>, tensor<1xf32>) -> tensor<10xf32>
// CHECK-NEXT:      return %2 : tensor<10xf32>
// CHECK-NEXT:    }

module {
  func.func @main(%arg0: tensor<3x4x5xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<4xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}, %arg2: tensor<3x4x5xf32> {enzymexla.memory_effects = []}, %arg3: tensor<4xf32> {enzymexla.memory_effects = []}) -> (tensor<3x4x5xf32>, tensor<4xf32>) attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<4> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0, %iterArg_5 = %arg1) : tensor<i64>, tensor<3x4x5xf32>, tensor<4xf32> attributes {enzymexla.disable_min_cut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c_0 : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg3, %3, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.remainder %iterArg, %c_2 : tensor<i64>
      %6 = stablehlo.add %5, %c_3 : tensor<i64>
      %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c_0 : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_5, %4, %8 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
      %10 = stablehlo.dynamic_slice %arg2, %c, %3, %c, sizes = [3, 1, 5] : (tensor<3x4x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x5xf32>
      %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %c, %3, %c : (tensor<3x4x5xf32>, tensor<3x1x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x5xf32>
      stablehlo.return %1, %11, %9 : tensor<i64>, tensor<3x4x5xf32>, tensor<4xf32>
    }
    return %0#1, %0#2 : tensor<3x4x5xf32>, tensor<4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x4x5xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<4xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}, %arg2: tensor<3x4x5xf32> {enzymexla.memory_effects = []}, %arg3: tensor<4xf32> {enzymexla.memory_effects = []}) -> (tensor<3x4x5xf32>, tensor<4xf32>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     return %arg2, %arg3 : tensor<3x4x5xf32>, tensor<4xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<11x9x7xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<9x6x4xf32> {enzymexla.memory_effects = []}) -> tensor<11x9x7xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %c_5 = stablehlo.constant dense<2> : tensor<i64>
    %c_6 = stablehlo.constant dense<5> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %arg0) : tensor<i64>, tensor<11x9x7xf32> attributes {enzymexla.disable_min_cut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %2 = stablehlo.add %iterArg, %c_4 : tensor<i64>
      %3 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_2 : tensor<i32>
      %5 = stablehlo.dynamic_slice %arg1, %c_0, %4, %c_1, sizes = [6, 1, 4] : (tensor<9x6x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x4xf32>
      %6 = stablehlo.dynamic_update_slice %iterArg_7, %5, %c_2, %4, %c : (tensor<11x9x7xf32>, tensor<6x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<11x9x7xf32>
      stablehlo.return %2, %6 : tensor<i64>, tensor<11x9x7xf32>
    }
    return %0#1 : tensor<11x9x7xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<11x9x7xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<9x6x4xf32> {enzymexla.memory_effects = []}) -> tensor<11x9x7xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %0 = stablehlo.slice %arg1 [3:9, 1:6, 0:4] : (tensor<9x6x4xf32>) -> tensor<6x5x4xf32>
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %c_0, %c : (tensor<11x9x7xf32>, tensor<6x5x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<11x9x7xf32>
// CHECK-NEXT:     return %1 : tensor<11x9x7xf32>
// CHECK-NEXT: }
