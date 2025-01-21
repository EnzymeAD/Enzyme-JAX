// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// CHECK-LABEL: @while_deadarg
func.func @while_deadarg(%arg0: tensor<2x6x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>, %arg5: tensor<2xui64>) -> (tensor<2x3xf32>, tensor<2xui64>, tensor<2x6x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>) {
  %c = stablehlo.constant dense<5> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x6x3xf32>) -> tensor<3x6x2xf32>
  %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %2 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %3 = stablehlo.slice %0 [0:3, 0:1, 0:2] : (tensor<3x6x2xf32>) -> tensor<3x1x2xf32>
  %4 = stablehlo.transpose %3, dims = [2, 1, 0] : (tensor<3x1x2xf32>) -> tensor<2x1x3xf32>
  %5 = stablehlo.reshape %4 : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  %6 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
  %7 = stablehlo.dot_general %arg1, %5, contracting_dims = [0] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
  %8 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
  %9 = stablehlo.add %7, %8 : tensor<3x2xf32>
  %10 = stablehlo.add %6, %9 : tensor<3x2xf32>
  %11 = stablehlo.tanh %10 : tensor<3x2xf32>
  %12 = stablehlo.reshape %11 : (tensor<3x2xf32>) -> tensor<3x2x1xf32>
  %13 = stablehlo.pad %12, %cst, low = [0, 0, 0], high = [0, 0, 5], interior = [0, 0, 0] : (tensor<3x2x1xf32>, tensor<f32>) -> tensor<3x2x6xf32>

  // CHECK: %{{.+}}:8 = stablehlo.while
  %14:9 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %13, %iterArg_4 = %1, %iterArg_5 = %2, %iterArg_6 = %arg3, %iterArg_7 = %arg4, %iterArg_8 = %arg5, %iterArg_9 = %11, %iterArg_10 = %0) : tensor<i64>, tensor<3x2x6xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x2xf32>, tensor<3x6x2xf32>
   cond {
    %19 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %19 : tensor<i1>
  } do {
    %19 = stablehlo.add %c_0, %iterArg : tensor<i64>
    %20 = stablehlo.subtract %19, %c_1 : tensor<i64>
    %21 = stablehlo.dynamic_slice %iterArg_10, %c_2, %20, %c_2, sizes = [3, 1, 2] : (tensor<3x6x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x1x2xf32>
    %22 = stablehlo.transpose %21, dims = [2, 1, 0] : (tensor<3x1x2xf32>) -> tensor<2x1x3xf32>
    %23 = stablehlo.reshape %22 : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
    %24 = stablehlo.dot_general %iterArg_5, %iterArg_9, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    %25 = stablehlo.broadcast_in_dim %iterArg_7, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
    %26 = stablehlo.add %24, %25 : tensor<3x2xf32>
    %27 = stablehlo.dot_general %iterArg_4, %23, contracting_dims = [1] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
    %28 = stablehlo.broadcast_in_dim %iterArg_6, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
    %29 = stablehlo.add %27, %28 : tensor<3x2xf32>
    %30 = stablehlo.add %26, %29 : tensor<3x2xf32>
    %31 = stablehlo.tanh %30 : tensor<3x2xf32>
    %32 = stablehlo.reshape %31 : (tensor<3x2xf32>) -> tensor<3x2x1xf32>
    // CHECK-NOT: dynamic_update_slice
    %33 = stablehlo.dynamic_update_slice %iterArg_3, %32, %c_2, %c_2, %20 : (tensor<3x2x6xf32>, tensor<3x2x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x2x6xf32>
    %34 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %34, %33, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %31, %iterArg_10 : tensor<i64>, tensor<3x2x6xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x2xf32>, tensor<3x6x2xf32>
  }
  %15 = stablehlo.transpose %14#7, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %16 = stablehlo.transpose %14#8, dims = [2, 1, 0] : (tensor<3x6x2xf32>) -> tensor<2x6x3xf32>
  %17 = stablehlo.transpose %14#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %18 = stablehlo.transpose %14#3, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %15, %14#6, %16, %17, %18, %14#4, %14#5 : tensor<2x3xf32>, tensor<2xui64>, tensor<2x6x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>
}
