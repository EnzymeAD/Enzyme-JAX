// RUN: enzymexlamlir-opt --enzyme-hlo-opt --inline %s | FileCheck %s

module @reactant_bar attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<5x7xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<7x5xf32>, tensor<5x7xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x7xf32>) -> tensor<7x5xf32>
    %1 = stablehlo.slice %0 [0:1, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %3 = stablehlo.reshape %2 : (tensor<5x1xf32>) -> tensor<5xf32>
    %4 = stablehlo.transpose %3, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %5 = "stablehlo.reduce_window"(%4, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %6 = stablehlo.slice %0 [1:2, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<5x1xf32>) -> tensor<5xf32>
    %9 = stablehlo.transpose %8, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %10 = "stablehlo.reduce_window"(%9, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %11 = stablehlo.slice %0 [2:3, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<5x1xf32>) -> tensor<5xf32>
    %14 = stablehlo.transpose %13, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %15 = "stablehlo.reduce_window"(%14, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %16 = stablehlo.slice %0 [3:4, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %17 = stablehlo.transpose %16, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %18 = stablehlo.reshape %17 : (tensor<5x1xf32>) -> tensor<5xf32>
    %19 = stablehlo.transpose %18, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %20 = "stablehlo.reduce_window"(%19, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %21 = stablehlo.slice %0 [4:5, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %22 = stablehlo.transpose %21, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<5x1xf32>) -> tensor<5xf32>
    %24 = stablehlo.transpose %23, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %25 = "stablehlo.reduce_window"(%24, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %26 = stablehlo.slice %0 [5:6, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %27 = stablehlo.transpose %26, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %28 = stablehlo.reshape %27 : (tensor<5x1xf32>) -> tensor<5xf32>
    %29 = stablehlo.transpose %28, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %30 = "stablehlo.reduce_window"(%29, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %31 = stablehlo.slice %0 [6:7, 0:5] : (tensor<7x5xf32>) -> tensor<1x5xf32>
    %32 = stablehlo.transpose %31, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %33 = stablehlo.reshape %32 : (tensor<5x1xf32>) -> tensor<5xf32>
    %34 = stablehlo.transpose %33, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %35 = "stablehlo.reduce_window"(%34, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %67 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %67 : tensor<f32>
    }) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
    %36 = stablehlo.transpose %5, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %37 = stablehlo.reshape %36 : (tensor<5xf32>) -> tensor<1x5xf32>
    %38 = stablehlo.transpose %37, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %39 = stablehlo.transpose %10, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %40 = stablehlo.reshape %39 : (tensor<5xf32>) -> tensor<1x5xf32>
    %41 = stablehlo.transpose %40, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %42 = stablehlo.transpose %15, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %43 = stablehlo.reshape %42 : (tensor<5xf32>) -> tensor<1x5xf32>
    %44 = stablehlo.transpose %43, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %45 = stablehlo.transpose %20, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %46 = stablehlo.reshape %45 : (tensor<5xf32>) -> tensor<1x5xf32>
    %47 = stablehlo.transpose %46, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %48 = stablehlo.transpose %25, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %49 = stablehlo.reshape %48 : (tensor<5xf32>) -> tensor<1x5xf32>
    %50 = stablehlo.transpose %49, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %51 = stablehlo.transpose %30, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %52 = stablehlo.reshape %51 : (tensor<5xf32>) -> tensor<1x5xf32>
    %53 = stablehlo.transpose %52, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %54 = stablehlo.transpose %35, dims = [0] : (tensor<5xf32>) -> tensor<5xf32>
    %55 = stablehlo.reshape %54 : (tensor<5xf32>) -> tensor<1x5xf32>
    %56 = stablehlo.transpose %55, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
    %57 = stablehlo.convert %38 : tensor<5x1xf32>
    %58 = stablehlo.convert %41 : tensor<5x1xf32>
    %59 = stablehlo.convert %44 : tensor<5x1xf32>
    %60 = stablehlo.convert %47 : tensor<5x1xf32>
    %61 = stablehlo.convert %50 : tensor<5x1xf32>
    %62 = stablehlo.convert %53 : tensor<5x1xf32>
    %63 = stablehlo.convert %56 : tensor<5x1xf32>
    %64 = stablehlo.concatenate %57, %58, %59, %60, %61, %62, %63, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x7xf32>
    %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<5x7xf32>) -> tensor<7x5xf32>
    %66 = stablehlo.transpose %0, dims = [1, 0] : (tensor<7x5xf32>) -> tensor<5x7xf32>
    return %65, %66 : tensor<7x5xf32>, tensor<5x7xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x7xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<7x5xf32>, tensor<5x7xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x7xf32>) -> tensor<7x5xf32>
// CHECK-NEXT{LITERAL}:     %1 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [4, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 5>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:       %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<7x5xf32>, tensor<f32>) -> tensor<7x5xf32>
// CHECK-NEXT:     return %1, %arg0 : tensor<7x5xf32>, tensor<5x7xf32>
// CHECK-NEXT: }
