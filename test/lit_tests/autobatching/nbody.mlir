// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<100x3xf32>, %arg1: tensor<100xf32>) -> tensor<100x100x3xf32> {
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<100> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<100x100x3xf32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<100x3xf32>) -> tensor<3x100xf32>
    %1:2 = stablehlo.while(%iterArg = %c_4, %iterArg_6 = %cst) : tensor<i64>, tensor<100x100x3xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_2 : tensor<i32>
      %5:2 = stablehlo.while(%iterArg_7 = %c_4, %iterArg_8 = %iterArg_6) : tensor<i64>, tensor<100x100x3xf32> attributes {enzyme.disable_mincut}
      cond {
        %6 = stablehlo.compare  LT, %iterArg_7, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %6 : tensor<i1>
      } do {
        %6 = stablehlo.add %c_5, %iterArg_7 : tensor<i64>
        %7 = stablehlo.dynamic_slice %0, %c_0, %4, sizes = [1, 1] : (tensor<3x100xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %8 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
        %9 = stablehlo.subtract %8, %c_2 : tensor<i32>
        %10 = stablehlo.dynamic_slice %0, %c_0, %9, sizes = [1, 1] : (tensor<3x100xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %11 = stablehlo.subtract %7, %10 : tensor<1x1xf32>
        %12 = stablehlo.reshape %11 : (tensor<1x1xf32>) -> tensor<f32>
        %13 = stablehlo.dynamic_slice %0, %c_2, %4, sizes = [1, 1] : (tensor<3x100xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %14 = stablehlo.dynamic_slice %0, %c_2, %9, sizes = [1, 1] : (tensor<3x100xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %15 = stablehlo.subtract %13, %14 : tensor<1x1xf32>
        %16 = stablehlo.dynamic_slice %0, %c, %4, sizes = [1, 1] : (tensor<3x100xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %17 = stablehlo.dynamic_slice %0, %c, %9, sizes = [1, 1] : (tensor<3x100xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %18 = stablehlo.subtract %16, %17 : tensor<1x1xf32>
        %19 = stablehlo.compare  EQ, %2, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %20 = stablehlo.multiply %12, %12 : tensor<f32>
        %21 = stablehlo.multiply %15, %15 : tensor<1x1xf32>
        %22 = stablehlo.reshape %21 : (tensor<1x1xf32>) -> tensor<f32>
        %23 = stablehlo.multiply %18, %18 : tensor<1x1xf32>
        %24 = stablehlo.reshape %23 : (tensor<1x1xf32>) -> tensor<f32>
        %25 = stablehlo.add %20, %22 : tensor<f32>
        %26 = stablehlo.add %25, %24 : tensor<f32>
        %27 = stablehlo.divide %cst_3, %26 : tensor<f32>
        %28 = stablehlo.select %19, %12, %27 : tensor<i1>, tensor<f32>
        %29 = stablehlo.dynamic_slice %arg1, %4, sizes = [1] : (tensor<100xf32>, tensor<i32>) -> tensor<1xf32>
        %30 = stablehlo.dynamic_slice %arg1, %9, sizes = [1] : (tensor<100xf32>, tensor<i32>) -> tensor<1xf32>
        %31 = stablehlo.multiply %29, %30 : tensor<1xf32>
        %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
        %33 = stablehlo.multiply %32, %28 : tensor<f32>
        %34 = stablehlo.reshape %33 : (tensor<f32>) -> tensor<1x1x1xf32>
        %35 = stablehlo.reshape %11 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %36 = stablehlo.reshape %15 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %37 = stablehlo.reshape %18 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %38 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<1x1x2xf32>
        %39 = stablehlo.concatenate %38, %34, dim = 2 : (tensor<1x1x2xf32>, tensor<1x1x1xf32>) -> tensor<1x1x3xf32>
        %40 = stablehlo.concatenate %35, %36, %37, dim = 2 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x3xf32>
        %41 = stablehlo.multiply %39, %40 : tensor<1x1x3xf32>
        %42 = stablehlo.dynamic_update_slice %iterArg_8, %41, %9, %4, %c_0 : (tensor<100x100x3xf32>, tensor<1x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<100x100x3xf32>
        stablehlo.return %6, %42 : tensor<i64>, tensor<100x100x3xf32>
      }
      stablehlo.return %2, %5#1 : tensor<i64>, tensor<100x100x3xf32>
    }
    return %1#1 : tensor<100x100x3xf32>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<100x3xf32>, %arg1: tensor<100xf32>) -> tensor<100x100x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<100x100xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<100x100xi64>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<100xf32>) -> tensor<100x100xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %arg0 [0:100, 2:3] : (tensor<100x3xf32>) -> tensor<100x1xf32>
// CHECK-NEXT:     %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<100x1xf32>) -> tensor<100x100x1x1xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %arg0 [0:100, 0:2] : (tensor<100x3xf32>) -> tensor<100x2xf32>
// CHECK-NEXT:     %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<100x2xf32>) -> tensor<2x100x100x1x1xf32>
// CHECK-NEXT:     %5 = stablehlo.iota dim = 1 : tensor<100x100xi64>
// CHECK-NEXT:     %6 = stablehlo.add %c, %5 : tensor<100x100xi64>
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %3, dims = [2, 0] : (tensor<100x2xf32>) -> tensor<2x100x100x1x1xf32>
// CHECK-NEXT:     %8 = stablehlo.slice %7 [0:1, 0:100, 0:100, 0:1, 0:1] : (tensor<2x100x100x1x1xf32>) -> tensor<1x100x100x1x1xf32>
// CHECK-NEXT:     %9 = stablehlo.reshape %8 : (tensor<1x100x100x1x1xf32>) -> tensor<100x100x1x1xf32>
// CHECK-NEXT:     %10 = stablehlo.slice %7 [1:2, 0:100, 0:100, 0:1, 0:1] : (tensor<2x100x100x1x1xf32>) -> tensor<1x100x100x1x1xf32>
// CHECK-NEXT:     %11 = stablehlo.reshape %10 : (tensor<1x100x100x1x1xf32>) -> tensor<100x100x1x1xf32>
// CHECK-NEXT:     %12 = stablehlo.concatenate %9, %11, dim = 0 : (tensor<100x100x1x1xf32>, tensor<100x100x1x1xf32>) -> tensor<200x100x1x1xf32>
// CHECK-NEXT:     %13 = stablehlo.reshape %12 : (tensor<200x100x1x1xf32>) -> tensor<2x100x100x1x1xf32>
// CHECK-NEXT:     %14 = stablehlo.subtract %4, %13 : tensor<2x100x100x1x1xf32>
// CHECK-NEXT:     %15 = stablehlo.slice %14 [0:1, 0:100, 0:100, 0:1, 0:1] : (tensor<2x100x100x1x1xf32>) -> tensor<1x100x100x1x1xf32>
// CHECK-NEXT:     %16 = stablehlo.slice %14 [1:2, 0:100, 0:100, 0:1, 0:1] : (tensor<2x100x100x1x1xf32>) -> tensor<1x100x100x1x1xf32>
// CHECK-NEXT:     %17 = stablehlo.reshape %16 : (tensor<1x100x100x1x1xf32>) -> tensor<100x100xf32>
// CHECK-NEXT:     %18 = stablehlo.iota dim = 0 : tensor<100x100xi64>
// CHECK-NEXT:     %19 = stablehlo.add %c, %18 : tensor<100x100xi64>
// CHECK-NEXT:     %20 = stablehlo.compare  EQ, %6, %19 : (tensor<100x100xi64>, tensor<100x100xi64>) -> tensor<100x100xi1>
// CHECK-NEXT:     %21 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<100x1xf32>) -> tensor<100x100x1x1xf32>
// CHECK-NEXT:     %22 = stablehlo.subtract %2, %21 : tensor<100x100x1x1xf32>
// CHECK-NEXT:     %23 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<100xf32>) -> tensor<100x100xf32>
// CHECK-NEXT:     %24 = stablehlo.multiply %0, %23 : tensor<100x100xf32>
// CHECK-NEXT:     %25 = stablehlo.reshape %22 : (tensor<100x100x1x1xf32>) -> tensor<100x100x1x1x1xf32>
// CHECK-NEXT:     %26 = stablehlo.transpose %14, dims = [1, 2, 3, 4, 0] : (tensor<2x100x100x1x1xf32>) -> tensor<100x100x1x1x2xf32>
// CHECK-NEXT:     %27 = stablehlo.concatenate %26, %25, dim = 4 : (tensor<100x100x1x1x2xf32>, tensor<100x100x1x1x1xf32>) -> tensor<100x100x1x1x3xf32>
// CHECK-NEXT:     %28 = stablehlo.multiply %22, %22 : tensor<100x100x1x1xf32>
// CHECK-NEXT:     %29 = stablehlo.multiply %17, %17 : tensor<100x100xf32>
// CHECK-NEXT:     %30 = stablehlo.reshape %15 : (tensor<1x100x100x1x1xf32>) -> tensor<100x100xf32>
// CHECK-NEXT:     %31 = stablehlo.multiply %30, %30 : tensor<100x100xf32>
// CHECK-NEXT:     %32 = stablehlo.add %31, %29 : tensor<100x100xf32>
// CHECK-NEXT:     %33 = stablehlo.reshape %28 : (tensor<100x100x1x1xf32>) -> tensor<100x100xf32>
// CHECK-NEXT:     %34 = stablehlo.add %32, %33 : tensor<100x100xf32>
// CHECK-NEXT:     %35 = stablehlo.divide %cst, %34 : tensor<100x100xf32>
// CHECK-NEXT:     %36 = stablehlo.select %20, %30, %35 : tensor<100x100xi1>, tensor<100x100xf32>
// CHECK-NEXT:     %37 = stablehlo.multiply %24, %36 : tensor<100x100xf32>
// CHECK-NEXT:     %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<100x100xf32>) -> tensor<100x100x1x1x2xf32>
// CHECK-NEXT:     %39 = stablehlo.reshape %37 : (tensor<100x100xf32>) -> tensor<100x100x1x1x1xf32>
// CHECK-NEXT:     %40 = stablehlo.concatenate %38, %39, dim = 4 : (tensor<100x100x1x1x2xf32>, tensor<100x100x1x1x1xf32>) -> tensor<100x100x1x1x3xf32>
// CHECK-NEXT:     %41 = stablehlo.multiply %40, %27 : tensor<100x100x1x1x3xf32>
// CHECK-NEXT:     %42 = stablehlo.reshape %41 : (tensor<100x100x1x1x3xf32>) -> tensor<100x100x3xf32>
// CHECK-NEXT:     %43 = stablehlo.transpose %42, dims = [1, 0, 2] : (tensor<100x100x3xf32>) -> tensor<100x100x3xf32>
// CHECK-NEXT:     return %43 : tensor<100x100x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
