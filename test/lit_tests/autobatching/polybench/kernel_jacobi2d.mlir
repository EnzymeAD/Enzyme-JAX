// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module @reactant_jacobi_2d attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<64x64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<64x64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}) -> (tensor<64x64xf32>, tensor<64x64xf32>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<2.000000e-01> : tensor<1x1xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<62> : tensor<i64>
    %c_4 = stablehlo.constant dense<32> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_5 = %arg0, %iterArg_6 = %arg1) : tensor<i64>, tensor<64x64xf32>, tensor<64x64xf32> attributes {enzyme.disable_mincut, enzymexla.symmetric_matrix = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed NOTGUARANTEED>]}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.transpose %iterArg_5, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
      %2 = stablehlo.transpose %iterArg_6, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
      %3 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %4:2 = stablehlo.while(%iterArg_7 = %c_1, %iterArg_8 = %2) : tensor<i64>, tensor<64x64xf32> attributes {enzyme.disable_mincut, enzymexla.symmetric_matrix = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>]}
      cond {
        %7 = stablehlo.compare  LT, %iterArg_7, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %7 : tensor<i1>
      } do {
        %7 = stablehlo.add %c_0, %iterArg_7 {enzymexla.bounds = [[2, 63]]} : tensor<i64>
        %8 = stablehlo.add %iterArg_7, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %9 = stablehlo.convert %7 {enzymexla.bounds = [[2, 63]]} : (tensor<i64>) -> tensor<i32>
        %10 = stablehlo.add %c_2, %7 {enzymexla.bounds = [[3, 64]]} : tensor<i64>
        %11 = stablehlo.subtract %7, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %12 = stablehlo.convert %11 {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
        %13 = stablehlo.convert %10 {enzymexla.bounds = [[3, 64]]} : (tensor<i64>) -> tensor<i32>
        %14 = stablehlo.subtract %9, %c {enzymexla.bounds = [[1, 62]]} : tensor<i32>
        %15 = stablehlo.subtract %12, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
        %16 = stablehlo.subtract %13, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
        %17:2 = stablehlo.while(%iterArg_9 = %c_1, %iterArg_10 = %iterArg_8) : tensor<i64>, tensor<64x64xf32> attributes {enzyme.disable_mincut}
        cond {
          %18 = stablehlo.compare  LT, %iterArg_9, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %18 : tensor<i1>
        } do {
          %18 = stablehlo.add %c_0, %iterArg_9 {enzymexla.bounds = [[2, 63]]} : tensor<i64>
          %19 = stablehlo.add %iterArg_9, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
          %20 = stablehlo.convert %18 {enzymexla.bounds = [[2, 63]]} : (tensor<i64>) -> tensor<i32>
          %21 = stablehlo.subtract %20, %c {enzymexla.bounds = [[1, 62]]} : tensor<i32>
          %22 = stablehlo.dynamic_slice %1, %14, %21, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %23 = stablehlo.subtract %18, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
          %24 = stablehlo.convert %23 {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
          %25 = stablehlo.subtract %24, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
          %26 = stablehlo.dynamic_slice %1, %14, %25, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %27 = stablehlo.add %c_2, %18 {enzymexla.bounds = [[3, 64]]} : tensor<i64>
          %28 = stablehlo.convert %27 {enzymexla.bounds = [[3, 64]]} : (tensor<i64>) -> tensor<i32>
          %29 = stablehlo.subtract %28, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
          %30 = stablehlo.dynamic_slice %1, %14, %29, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %31 = stablehlo.dynamic_slice %1, %16, %21, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %32 = stablehlo.dynamic_slice %1, %15, %21, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %33 = stablehlo.add %22, %26 : tensor<1x1xf32>
          %34 = stablehlo.add %33, %30 : tensor<1x1xf32>
          %35 = stablehlo.add %34, %31 : tensor<1x1xf32>
          %36 = stablehlo.add %35, %32 : tensor<1x1xf32>
          %37 = stablehlo.multiply %cst, %36 : tensor<1x1xf32>
          %38 = stablehlo.dynamic_update_slice %iterArg_10, %37, %14, %21 : (tensor<64x64xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<64x64xf32>
          stablehlo.return %19, %38 : tensor<i64>, tensor<64x64xf32>
        }
        stablehlo.return %8, %17#1 : tensor<i64>, tensor<64x64xf32>
      }
      %5:2 = stablehlo.while(%iterArg_7 = %c_1, %iterArg_8 = %iterArg_5) : tensor<i64>, tensor<64x64xf32> attributes {enzyme.disable_mincut}
      cond {
        %7 = stablehlo.compare  LT, %iterArg_7, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %7 : tensor<i1>
      } do {
        %7 = stablehlo.add %c_0, %iterArg_7 {enzymexla.bounds = [[2, 63]]} : tensor<i64>
        %8 = stablehlo.add %iterArg_7, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %9 = stablehlo.convert %7 {enzymexla.bounds = [[2, 63]]} : (tensor<i64>) -> tensor<i32>
        %10 = stablehlo.add %c_2, %7 {enzymexla.bounds = [[3, 64]]} : tensor<i64>
        %11 = stablehlo.subtract %7, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
        %12 = stablehlo.convert %11 {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
        %13 = stablehlo.convert %10 {enzymexla.bounds = [[3, 64]]} : (tensor<i64>) -> tensor<i32>
        %14 = stablehlo.subtract %9, %c {enzymexla.bounds = [[1, 62]]} : tensor<i32>
        %15 = stablehlo.subtract %12, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
        %16 = stablehlo.subtract %13, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
        %17:2 = stablehlo.while(%iterArg_9 = %c_1, %iterArg_10 = %iterArg_8) : tensor<i64>, tensor<64x64xf32> attributes {enzyme.disable_mincut}
        cond {
          %18 = stablehlo.compare  LT, %iterArg_9, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %18 : tensor<i1>
        } do {
          %18 = stablehlo.add %c_0, %iterArg_9 {enzymexla.bounds = [[2, 63]]} : tensor<i64>
          %19 = stablehlo.add %iterArg_9, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
          %20 = stablehlo.convert %18 {enzymexla.bounds = [[2, 63]]} : (tensor<i64>) -> tensor<i32>
          %21 = stablehlo.subtract %20, %c {enzymexla.bounds = [[1, 62]]} : tensor<i32>
          %22 = stablehlo.dynamic_slice %4#1, %14, %21, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %23 = stablehlo.subtract %18, %c_2 {enzymexla.bounds = [[1, 62]]} : tensor<i64>
          %24 = stablehlo.convert %23 {enzymexla.bounds = [[1, 62]]} : (tensor<i64>) -> tensor<i32>
          %25 = stablehlo.subtract %24, %c {enzymexla.bounds = [[0, 61]]} : tensor<i32>
          %26 = stablehlo.dynamic_slice %4#1, %14, %25, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %27 = stablehlo.add %c_2, %18 {enzymexla.bounds = [[3, 64]]} : tensor<i64>
          %28 = stablehlo.convert %27 {enzymexla.bounds = [[3, 64]]} : (tensor<i64>) -> tensor<i32>
          %29 = stablehlo.subtract %28, %c {enzymexla.bounds = [[2, 63]]} : tensor<i32>
          %30 = stablehlo.dynamic_slice %4#1, %14, %29, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %31 = stablehlo.dynamic_slice %4#1, %16, %21, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %32 = stablehlo.dynamic_slice %4#1, %15, %21, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
          %33 = stablehlo.add %22, %26 : tensor<1x1xf32>
          %34 = stablehlo.add %33, %30 : tensor<1x1xf32>
          %35 = stablehlo.add %34, %31 : tensor<1x1xf32>
          %36 = stablehlo.add %35, %32 : tensor<1x1xf32>
          %37 = stablehlo.multiply %cst, %36 : tensor<1x1xf32>
          %38 = stablehlo.dynamic_update_slice %iterArg_10, %37, %21, %14 : (tensor<64x64xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<64x64xf32>
          stablehlo.return %19, %38 : tensor<i64>, tensor<64x64xf32>
        }
        stablehlo.return %8, %17#1 : tensor<i64>, tensor<64x64xf32>
      }
      %6 = stablehlo.transpose %4#1, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
      stablehlo.return %3, %5#1, %6 : tensor<i64>, tensor<64x64xf32>, tensor<64x64xf32>
    }
    return %0#1, %0#2 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}

// CHECK: %5 = stablehlo.dynamic_update_slice %arg0, %iterArg_4, %c, %c : (tensor<64x64xf32>, tensor<62x62xf32>, tensor<i32>, tensor<i32>) -> tensor<64x64xf32>
// CHECK-NEXT: %6 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT: %7 = stablehlo.slice %5 [0:64, 1:63] : (tensor<64x64xf32>) -> tensor<64x62xf32>
// CHECK-NEXT: %8 = stablehlo.broadcast_in_dim %7, dims = [2, 0] : (tensor<64x62xf32>) -> tensor<62x1x64xf32>
// CHECK-NEXT: %9 = "stablehlo.reduce_window"(%8, %cst_0) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %32 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %32 : tensor<f32>
// CHECK-NEXT: }) : (tensor<62x1x64xf32>, tensor<f32>) -> tensor<62x1x62xf32>
// CHECK-NEXT: %10 = stablehlo.reshape %9 : (tensor<62x1x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %11 = stablehlo.slice %5 [1:63, 2:64] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<62x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %13 = stablehlo.add %10, %12 : tensor<62x62xf32>
// CHECK-NEXT: %14 = stablehlo.slice %5 [1:63, 0:62] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<62x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %16 = stablehlo.add %13, %15 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<62x62xf32>
// CHECK-NEXT: %17 = stablehlo.multiply %cst, %16 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<62x62xf32>
// CHECK-NEXT: %18 = stablehlo.transpose %17, dims = [1, 0] : (tensor<62x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %19 = stablehlo.dynamic_update_slice %arg1, %18, %c, %c {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64xf32>, tensor<62x62xf32>, tensor<i32>, tensor<i32>) -> tensor<64x64xf32>
// CHECK-NEXT: %20 = stablehlo.slice %19 [0:64, 1:63] : (tensor<64x64xf32>) -> tensor<64x62xf32>
// CHECK-NEXT: %21 = stablehlo.broadcast_in_dim %20, dims = [2, 0] : (tensor<64x62xf32>) -> tensor<62x1x64xf32>
// CHECK-NEXT: %22 = "stablehlo.reduce_window"(%21, %cst_0) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3>, window_strides = array<i64: 1, 1, 1>}> ({
// CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:   %32 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:   stablehlo.return %32 : tensor<f32>
// CHECK-NEXT: }) : (tensor<62x1x64xf32>, tensor<f32>) -> tensor<62x1x62xf32>
// CHECK-NEXT: %23 = stablehlo.reshape %22 : (tensor<62x1x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %24 = stablehlo.slice %19 [1:63, 2:64] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %25 = stablehlo.transpose %24, dims = [1, 0] : (tensor<62x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %26 = stablehlo.add %23, %25 : tensor<62x62xf32>
// CHECK-NEXT: %27 = stablehlo.slice %19 [1:63, 0:62] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<64x64xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %28 = stablehlo.transpose %27, dims = [1, 0] : (tensor<62x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: %29 = stablehlo.add %26, %28 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<62x62xf32>
// CHECK-NEXT: %30 = stablehlo.multiply %cst, %29 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<62x62xf32>
// CHECK-NEXT: %31 = stablehlo.transpose %30, dims = [1, 0] : (tensor<62x62xf32>) -> tensor<62x62xf32>
// CHECK-NEXT: stablehlo.return %6, %31, %18 : tensor<i64>, tensor<62x62xf32>, tensor<62x62xf32>
