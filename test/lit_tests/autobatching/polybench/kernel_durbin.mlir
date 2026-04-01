// RUN: enzymexlamlir-opt   --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module @reactant_kernel_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<64xf32> {enzymexla.memory_effects = []}, %arg1: tensor<64xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<64xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<63> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_1 = stablehlo.constant dense<64> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i32>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<64xf32>) -> tensor<1xf32>
    %1 = stablehlo.negate %0 : tensor<1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.slice %arg1 [1:64] : (tensor<64xf32>) -> tensor<63xf32>
    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1xf32>, tensor<63xf32>) -> tensor<64xf32>
    // CHECK: stablehlo.while
    %5:5 = stablehlo.while(%iterArg = %c_3, %iterArg_6 = %2, %iterArg_7 = %cst_0, %iterArg_8 = %4, %iterArg_9 = %cst) : tensor<i64>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32> attributes {enzyme.disable_mincut}
    cond {
      %6 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    } do {
      %6 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = [[2, 64]]} : tensor<i64>
      %7 = stablehlo.add %iterArg, %c_5 {enzymexla.bounds = [[1, 63]]} : tensor<i64>
      %8 = stablehlo.multiply %iterArg_6, %iterArg_6 : tensor<f32>
      %9 = stablehlo.subtract %cst_0, %8 : tensor<f32>
      %10 = stablehlo.multiply %9, %iterArg_7 : tensor<f32>
      %11:2 = stablehlo.while(%iterArg_10 = %c_3, %iterArg_11 = %cst_0) : tensor<i64>, tensor<f32> attributes {enzyme.disable_mincut}
      cond {
        %23 = stablehlo.compare  LT, %iterArg_10, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %23 : tensor<i1>
      } do {
        %23 = stablehlo.add %c_5, %iterArg_10 : tensor<i64>
        %24 = stablehlo.subtract %6, %23 : tensor<i64>
        %25 = stablehlo.convert %24 : (tensor<i64>) -> tensor<i32>
        %26 = stablehlo.subtract %25, %c_4 : tensor<i32>
        %27 = stablehlo.dynamic_slice %arg0, %26, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %28 = stablehlo.convert %23 : (tensor<i64>) -> tensor<i32>
        %29 = stablehlo.subtract %28, %c_4 : tensor<i32>
        %30 = stablehlo.dynamic_slice %iterArg_8, %29, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %31 = stablehlo.multiply %27, %30 : tensor<1xf32>
        %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
        %33 = stablehlo.add %iterArg_11, %32 : tensor<f32>
        stablehlo.return %23, %33 : tensor<i64>, tensor<f32>
      }
      %12 = stablehlo.convert %6 {enzymexla.bounds = [[2, 64]]} : (tensor<i64>) -> tensor<i32>
      %13 = stablehlo.subtract %12, %c_4 {enzymexla.bounds = [[1, 63]]} : tensor<i32>
      %14 = stablehlo.dynamic_slice %arg0, %13, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
      %15 = stablehlo.reshape %14 : (tensor<1xf32>) -> tensor<f32>
      %16 = stablehlo.add %15, %11#1 : tensor<f32>
      %17 = stablehlo.negate %16 : tensor<f32>
      %18 = stablehlo.divide %17, %10 : tensor<f32>
      %19 = stablehlo.reshape %18 : (tensor<f32>) -> tensor<1xf32>
      %20:2 = stablehlo.while(%iterArg_10 = %c_3, %iterArg_11 = %iterArg_9) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %23 = stablehlo.compare  LT, %iterArg_10, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %23 : tensor<i1>
      } do {
        %23 = stablehlo.add %c_5, %iterArg_10 : tensor<i64>
        %24 = stablehlo.convert %23 : (tensor<i64>) -> tensor<i32>
        %25 = stablehlo.subtract %24, %c_4 : tensor<i32>
        %26 = stablehlo.dynamic_slice %iterArg_8, %25, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %27 = stablehlo.subtract %6, %23 : tensor<i64>
        %28 = stablehlo.convert %27 : (tensor<i64>) -> tensor<i32>
        %29 = stablehlo.subtract %28, %c_4 : tensor<i32>
        %30 = stablehlo.dynamic_slice %iterArg_8, %29, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %31 = stablehlo.multiply %19, %30 : tensor<1xf32>
        %32 = stablehlo.add %26, %31 : tensor<1xf32>
        %33 = stablehlo.remainder %iterArg_10, %c_1 : tensor<i64>
        %34 = stablehlo.add %33, %c_5 : tensor<i64>
        %35 = stablehlo.convert %34 : (tensor<i64>) -> tensor<i32>
        %36 = stablehlo.subtract %35, %c_4 : tensor<i32>
        %37 = stablehlo.dynamic_update_slice %iterArg_11, %32, %36 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %23, %37 : tensor<i64>, tensor<64xf32>
      }
      %21:2 = stablehlo.while(%iterArg_10 = %c_3, %iterArg_11 = %iterArg_8) : tensor<i64>, tensor<64xf32> attributes {enzyme.disable_mincut}
      cond {
        %23 = stablehlo.compare  LT, %iterArg_10, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %23 : tensor<i1>
      } do {
        %23 = stablehlo.add %c_5, %iterArg_10 : tensor<i64>
        %24 = stablehlo.convert %23 : (tensor<i64>) -> tensor<i32>
        %25 = stablehlo.subtract %24, %c_4 : tensor<i32>
        %26 = stablehlo.dynamic_slice %20#1, %25, sizes = [1] : (tensor<64xf32>, tensor<i32>) -> tensor<1xf32>
        %27 = stablehlo.remainder %iterArg_10, %c_1 : tensor<i64>
        %28 = stablehlo.add %27, %c_5 : tensor<i64>
        %29 = stablehlo.convert %28 : (tensor<i64>) -> tensor<i32>
        %30 = stablehlo.subtract %29, %c_4 : tensor<i32>
        %31 = stablehlo.dynamic_update_slice %iterArg_11, %26, %30 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
        stablehlo.return %23, %31 : tensor<i64>, tensor<64xf32>
      }
      %22 = stablehlo.dynamic_update_slice %21#1, %19, %13 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
      stablehlo.return %7, %18, %10, %22, %20#1 : tensor<i64>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>
    }
    return %5#3 : tensor<64xf32>
  }
}
