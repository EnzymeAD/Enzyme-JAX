// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module @reactant_sine_lo... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
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
      %4 = stablehlo.dynamic_slice %iterArg_3, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.sine %4 : tensor<1xf32>
      %6 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %7 = stablehlo.add %6, %c_2 : tensor<i64>
      %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c : tensor<i32>
      %10 = stablehlo.dynamic_update_slice %iterArg_3, %5, %9 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %10 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzymexla.disable_min_cut}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:       %3 = stablehlo.subtract %2, %c : tensor<i32>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %iterArg_3, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:       %5 = stablehlo.sine %4 : tensor<1xf32>
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %3 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK-NEXT:       stablehlo.return %1, %6 : tensor<i64>, tensor<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<10xf32>
// CHECK-NEXT: }
