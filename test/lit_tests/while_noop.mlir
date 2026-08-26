// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
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
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.convert %1 {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:       %3 = stablehlo.subtract %2, %c {enzymexla.bounds = {{\ *\[\[}}0, 9{{\ *\]\]}}{{.*}}} : tensor<i32>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %iterArg_3, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:       %5 = stablehlo.sine %4 : tensor<1xf32>
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %3 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK-NEXT:       stablehlo.return %1, %6 : tensor<i64>, tensor<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<10xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<31> : tensor<i64>
    %c_1 = stablehlo.constant dense<21> : tensor<i64>
    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
    %c_3 = stablehlo.constant dense<2> : tensor<i64>
    %c_4 = stablehlo.constant dense<5> : tensor<i64>
    %c_5 = stablehlo.constant dense<4> : tensor<i64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_7 = stablehlo.constant dense<10> : tensor<i64>
    %c_8 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_6, %iterArg_9 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_8, %iterArg : tensor<i64>
      %2 = stablehlo.add %c_5, %1 : tensor<i64>
      %3 = stablehlo.subtract %2, %c_4 : tensor<i64>
      %4 = stablehlo.abs %3 : tensor<i64>
      %5 = stablehlo.multiply %4, %c_3 : tensor<i64>
      %6 = stablehlo.clamp %c_2, %5, %c_1 : tensor<i64>
      %7 = stablehlo.compare  GT, %6, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %8 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c : tensor<i32>
      %10 = stablehlo.dynamic_slice %iterArg_9, %9, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %11 = stablehlo.reshape %10 : (tensor<1xf32>) -> tensor<f32>
      %12 = stablehlo.sine %11 : tensor<f32>
      %13 = stablehlo.cosine %11 : tensor<f32>
      %14 = stablehlo.select %7, %12, %13 : tensor<i1>, tensor<f32>
      %15 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1xf32>
      %16 = stablehlo.remainder %iterArg, %c_7 : tensor<i64>
      %17 = stablehlo.add %16, %c_8 : tensor<i64>
      %18 = stablehlo.convert %17 : (tensor<i64>) -> tensor<i32>
      %19 = stablehlo.subtract %18, %c : tensor<i32>
      %20 = stablehlo.dynamic_update_slice %iterArg_9, %15, %19 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %20 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.convert %1 {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:       %3 = stablehlo.subtract %2, %c {enzymexla.bounds = {{\ *\[\[}}0, 9{{\ *\]\]}}{{.*}}} : tensor<i32>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %iterArg_3, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:       %5 = stablehlo.cosine %4 : tensor<1xf32>
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %3 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK-NEXT:       stablehlo.return %1, %6 : tensor<i64>, tensor<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

module @reactant_sine_lo... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<31> : tensor<i64>
    %c_1 = stablehlo.constant dense<21> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<5> : tensor<i64>
    %c_4 = stablehlo.constant dense<4> : tensor<i64>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %c_6 = stablehlo.constant dense<-1> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i64>
    %c_8 = stablehlo.constant dense<10> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_5, %iterArg_9 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.subtract %c_8, %iterArg : tensor<i64>
      %2 = stablehlo.add %iterArg, %c_7 : tensor<i64>
      %3 = stablehlo.add %c_4, %1 : tensor<i64>
      %4 = stablehlo.subtract %3, %c_3 : tensor<i64>
      %5 = stablehlo.abs %4 : tensor<i64>
      %6 = stablehlo.multiply %5, %c_2 : tensor<i64>
      %7 = stablehlo.clamp %c_6, %6, %c_1 : tensor<i64>
      %8 = stablehlo.compare  GT, %7, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %10 = stablehlo.subtract %9, %c : tensor<i32>
      %11 = stablehlo.dynamic_slice %iterArg_9, %10, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %12 = stablehlo.reshape %11 : (tensor<1xf32>) -> tensor<f32>
      %13 = stablehlo.sine %12 : tensor<f32>
      %14 = stablehlo.cosine %12 : tensor<f32>
      %15 = stablehlo.select %8, %13, %14 : tensor<i1>, tensor<f32>
      %16 = stablehlo.reshape %15 : (tensor<f32>) -> tensor<1xf32>
      %17 = stablehlo.subtract %1, %c_7 : tensor<i64>
      %18 = stablehlo.remainder %17, %c_8 : tensor<i64>
      %19 = stablehlo.add %18, %c_7 : tensor<i64>
      %20 = stablehlo.convert %19 : (tensor<i64>) -> tensor<i32>
      %21 = stablehlo.subtract %20, %c : tensor<i32>
      %22 = stablehlo.dynamic_update_slice %iterArg_9, %16, %21 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %2, %22 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: module @reactant_sine_lo... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:     func.func @main(%arg0: tensor<10xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:       %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:       %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:       %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:       %c_2 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:       %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32> attributes {enzyme.disable_mincut}
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %1 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %1 = stablehlo.subtract %c_2, %iterArg {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : tensor<i64>
// CHECK-NEXT:         %2 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : tensor<i64>
// CHECK-NEXT:         %3 = stablehlo.convert %1 {enzymexla.bounds = {{\ *\[\[}}1, 10{{\ *\]\]}}{{.*}}} : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:         %4 = stablehlo.subtract %3, %c {enzymexla.bounds = {{\ *\[\[}}0, 9{{\ *\]\]}}{{.*}}} : tensor<i32>
// CHECK-NEXT:         %5 = stablehlo.dynamic_slice %iterArg_3, %4, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:         %6 = stablehlo.cosine %5 : tensor<1xf32>
// CHECK-NEXT:         %7 = stablehlo.dynamic_update_slice %iterArg_3, %6, %4 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK-NEXT:         stablehlo.return %2, %7 : tensor<i64>, tensor<10xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       return %0#1 : tensor<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT: }

