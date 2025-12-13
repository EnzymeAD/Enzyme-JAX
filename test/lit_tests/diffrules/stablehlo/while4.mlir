// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --remove-unnecessary-enzyme-ops | FileCheck %s

module {
  func.func private @"diffeConst{typeof(sumabs2)}(Main.sumabs2)_autodiff"(%arg0: tensor<6x2x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>, %arg5: tensor<2xui64>, %arg6: tensor<f32>, %arg7: tensor<2xui64>, %arg8: tensor<3x3xf32>, %arg9: tensor<3x3xf32>, %arg10: tensor<3xf32>, %arg11: tensor<3xf32>) -> (tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x2xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<6> : tensor<i64>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf32>
    %0 = "enzyme.init"() : () -> !enzyme.Cache<tensor<3x2xf32>>
    %1 = "enzyme.init"() : () -> !enzyme.Cache<tensor<3x3xf32>>
    %2 = "enzyme.init"() : () -> !enzyme.Cache<tensor<2x3xf32>>
    %3 = "enzyme.init"() : () -> !enzyme.Cache<tensor<3x2xf32>>
    %4 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<6x2x3xf32>) -> tensor<3x2x6xf32>
    %5 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %6 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %7 = stablehlo.slice %4 [0:3, 0:2, 0:1] : (tensor<3x2x6xf32>) -> tensor<3x2x1xf32>
    %8 = stablehlo.transpose %7, dims = [2, 1, 0] : (tensor<3x2x1xf32>) -> tensor<1x2x3xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
    %10 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
    %11 = stablehlo.dot_general %arg1, %9, contracting_dims = [0] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
    %13 = stablehlo.add %11, %12 : tensor<3x2xf32>
    %14 = stablehlo.add %10, %13 : tensor<3x2xf32>
    %15 = stablehlo.tanh %14 : tensor<3x2xf32>
    %16:10 = stablehlo.while(%iterArg = %c, %iterArg_5 = %5, %iterArg_6 = %6, %iterArg_7 = %arg3, %iterArg_8 = %arg4, %iterArg_9 = %arg5, %iterArg_10 = %c_0, %iterArg_11 = %15, %iterArg_12 = %4, %iterArg_13 = %c) : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<i64>, tensor<3x2xf32>, tensor<3x2x6xf32>, tensor<i64>
     cond {
      %41 = stablehlo.subtract %iterArg_10, %c_1 : tensor<i64>
      %42 = stablehlo.divide %41, %c_2 : tensor<i64>
      %43 = stablehlo.add %42, %c_2 : tensor<i64>
      %44 = stablehlo.compare  LT, %iterArg, %43 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %44 : tensor<i1>
    } do {
      %41 = stablehlo.add %iterArg_13, %c_2 : tensor<i64>
      %42 = stablehlo.add %c_1, %iterArg : tensor<i64>
      %43 = stablehlo.subtract %42, %c_2 : tensor<i64>
      %44 = stablehlo.dynamic_slice %iterArg_12, %c, %c, %43, sizes = [3, 2, 1] : (tensor<3x2x6xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x2x1xf32>
      %45 = stablehlo.transpose %44, dims = [2, 1, 0] : (tensor<3x2x1xf32>) -> tensor<1x2x3xf32>
      %46 = stablehlo.reshape %45 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
      "enzyme.push"(%1, %iterArg_6) : (!enzyme.Cache<tensor<3x3xf32>>, tensor<3x3xf32>) -> ()
      "enzyme.push"(%0, %iterArg_11) : (!enzyme.Cache<tensor<3x2xf32>>, tensor<3x2xf32>) -> ()
      %47 = stablehlo.dot_general %iterArg_6, %iterArg_11, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
      %48 = stablehlo.broadcast_in_dim %iterArg_8, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
      %49 = stablehlo.add %47, %48 : tensor<3x2xf32>
      "enzyme.push"(%2, %46) : (!enzyme.Cache<tensor<2x3xf32>>, tensor<2x3xf32>) -> ()
      %50 = stablehlo.dot_general %iterArg_5, %46, contracting_dims = [1] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
      %51 = stablehlo.broadcast_in_dim %iterArg_7, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
      %52 = stablehlo.add %50, %51 : tensor<3x2xf32>
      %53 = stablehlo.add %49, %52 : tensor<3x2xf32>
      "enzyme.push"(%3, %53) : (!enzyme.Cache<tensor<3x2xf32>>, tensor<3x2xf32>) -> ()
      %54 = stablehlo.tanh %53 : tensor<3x2xf32>
      %55 = stablehlo.add %iterArg, %c_2 : tensor<i64>
      stablehlo.return %55, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %54, %iterArg_12, %41 : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<i64>, tensor<3x2xf32>, tensor<3x2x6xf32>, tensor<i64>
    }
    %17 = stablehlo.abs %16#7 : tensor<3x2xf32>
    %18 = stablehlo.transpose %16#8, dims = [2, 1, 0] : (tensor<3x2x6xf32>) -> tensor<6x2x3xf32>
    %19 = stablehlo.transpose %16#1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %20 = stablehlo.transpose %16#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %21 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %22 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %23 = stablehlo.broadcast_in_dim %arg6, dims = [] : (tensor<f32>) -> tensor<3x2xf32>
    %24 = stablehlo.multiply %23, %17 : tensor<3x2xf32>
    %25 = stablehlo.add %24, %24 : tensor<3x2xf32>
    %26 = stablehlo.compare  GE, %16#7, %cst_4 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xi1>
    %27 = stablehlo.negate %25 : tensor<3x2xf32>
    %28 = stablehlo.select %26, %25, %27 : tensor<3x2xi1>, tensor<3x2xf32>
    %29:6 = stablehlo.while(%iterArg = %c, %iterArg_5 = %22, %iterArg_6 = %21, %iterArg_7 = %arg10, %iterArg_8 = %arg11, %iterArg_9 = %28) : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>
     cond {
      %41 = stablehlo.compare  LT, %iterArg, %16#9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %41 : tensor<i1>
    } do {
      %41 = stablehlo.add %iterArg, %c_2 : tensor<i64>
      %42 = "enzyme.pop"(%3) : (!enzyme.Cache<tensor<3x2xf32>>) -> tensor<3x2xf32>
      %43 = stablehlo.tanh %42 : tensor<3x2xf32>
      %44 = stablehlo.multiply %43, %43 : tensor<3x2xf32>
      %45 = stablehlo.subtract %cst, %44 : tensor<3x2xf32>
      %46 = stablehlo.multiply %iterArg_9, %45 : tensor<3x2xf32>
      %47 = stablehlo.reduce(%46 init: %cst_3) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
      %48 = stablehlo.add %iterArg_7, %47 : tensor<3xf32>
      %49 = "enzyme.pop"(%2) : (!enzyme.Cache<tensor<2x3xf32>>) -> tensor<2x3xf32>
      %50 = stablehlo.dot_general %46, %49, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
      %51 = stablehlo.add %iterArg_5, %50 : tensor<3x3xf32>
      %52 = stablehlo.add %iterArg_8, %47 : tensor<3xf32>
      %53 = "enzyme.pop"(%1) : (!enzyme.Cache<tensor<3x3xf32>>) -> tensor<3x3xf32>
      %54 = "enzyme.pop"(%0) : (!enzyme.Cache<tensor<3x2xf32>>) -> tensor<3x2xf32>
      %55 = stablehlo.dot_general %46, %54, contracting_dims = [1] x [1] : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x3xf32>
      %56 = stablehlo.add %iterArg_6, %55 : tensor<3x3xf32>
      %57 = stablehlo.dot_general %53, %46, contracting_dims = [0] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
      stablehlo.return %41, %51, %56, %48, %52, %57 : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>
    }
    %30 = stablehlo.multiply %15, %15 : tensor<3x2xf32>
    %31 = stablehlo.subtract %cst, %30 : tensor<3x2xf32>
    %32 = stablehlo.multiply %29#5, %31 : tensor<3x2xf32>
    %33 = stablehlo.reduce(%32 init: %cst_3) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
    %34 = stablehlo.add %29#3, %33 : tensor<3xf32>
    %35 = stablehlo.dot_general %32, %9, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
    %36 = stablehlo.reduce(%32 init: %cst_3) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
    %37 = stablehlo.add %29#4, %36 : tensor<3xf32>
    %38 = stablehlo.transpose %29#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %39 = stablehlo.add %35, %29#1 : tensor<3x3xf32>
    %40 = stablehlo.transpose %39, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %18, %19, %20, %16#3, %16#4, %arg5, %40, %38, %34, %37 : tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>
  }
  func.func @main(%arg0: tensor<6x2x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>, %arg5: tensor<2xui64>) -> (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>) {
    %c = stablehlo.constant dense<1> : tensor<2xui64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
    %0:10 = call @"diffeConst{typeof(sumabs2)}(Main.sumabs2)_autodiff"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %cst, %c, %cst_1, %cst_1, %cst_0, %cst_0) : (tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<f32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>)
    return %0#6, %0#7, %0#8, %0#9, %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>
  }
}

// CHECK:  func.func private @"diffeConst{typeof(sumabs2)}(Main.sumabs2)_autodiff"(%arg0: tensor<6x2x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>, %arg5: tensor<2xui64>, %arg6: tensor<f32>, %arg7: tensor<2xui64>, %arg8: tensor<3x3xf32>, %arg9: tensor<3x3xf32>, %arg10: tensor<3xf32>, %arg11: tensor<3xf32>) -> (tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<5x3x2x1xf32>
// CHECK-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<5x3x2xf32>
// CHECK-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<5x3x3xf32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<3x2xf32>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf32>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<6x2x3xf32>) -> tensor<3x2x6xf32>
// CHECK-NEXT:    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %2 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [0:3, 0:2, 0:1] : (tensor<3x2x6xf32>) -> tensor<3x2x1xf32>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [2, 1, 0] : (tensor<3x2x1xf32>) -> tensor<1x2x3xf32>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:    %7 = stablehlo.dot_general %arg1, %5, contracting_dims = [0] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:    %9 = stablehlo.add %7, %8 : tensor<3x2xf32>
// CHECK-NEXT:    %10 = stablehlo.add %6, %9 : tensor<3x2xf32>
// CHECK-NEXT:    %11 = stablehlo.tanh %10 : tensor<3x2xf32>
// CHECK-NEXT:    %12:9 = stablehlo.while(%iterArg = %c_4, %iterArg_8 = %1, %iterArg_9 = %2, %iterArg_10 = %11, %iterArg_11 = %0, %iterArg_12 = %cst_1, %iterArg_13 = %cst_0, %iterArg_14 = %cst_0, %iterArg_15 = %cst) : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x2xf32>, tensor<3x2x6xf32>, tensor<5x3x3xf32>, tensor<5x3x2xf32>, tensor<5x3x2xf32>, tensor<5x3x2x1xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %36 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %36 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %36 = stablehlo.reshape %iterArg_10 : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %37 = stablehlo.dynamic_update_slice %iterArg_13, %36, %iterArg, %c_4, %c_4 : (tensor<5x3x2xf32>, tensor<1x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x2xf32>
// CHECK-NEXT:      %38 = stablehlo.reshape %iterArg_9 : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK-NEXT:      %39 = stablehlo.dynamic_update_slice %iterArg_12, %38, %iterArg, %c_4, %c_4 : (tensor<5x3x3xf32>, tensor<1x3x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x3xf32>
// CHECK-NEXT:      %40 = stablehlo.add %iterArg, %c_6 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      %41 = stablehlo.add %c_5, %iterArg {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      %42 = stablehlo.subtract %41, %c_6 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      %43 = stablehlo.dynamic_slice %iterArg_11, %c_4, %c_4, %42, sizes = [3, 2, 1] : (tensor<3x2x6xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x2x1xf32>
// CHECK-NEXT:      %44 = stablehlo.reshape %43 : (tensor<3x2x1xf32>) -> tensor<1x3x2x1xf32>
// CHECK-NEXT:      %45 = stablehlo.dynamic_update_slice %iterArg_15, %44, %iterArg, %c_4, %c_4, %c_4 : (tensor<5x3x2x1xf32>, tensor<1x3x2x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x2x1xf32>
// CHECK-NEXT:      %46 = stablehlo.transpose %43, dims = [2, 1, 0] : (tensor<3x2x1xf32>) -> tensor<1x2x3xf32>
// CHECK-NEXT:      %47 = stablehlo.reshape %46 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:      %48 = stablehlo.dot_general %iterArg_9, %iterArg_10, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %49 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %50 = stablehlo.add %48, %49 : tensor<3x2xf32>
// CHECK-NEXT:      %51 = stablehlo.dot_general %iterArg_8, %47, contracting_dims = [1] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %52 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %53 = stablehlo.add %51, %52 : tensor<3x2xf32>
// CHECK-NEXT:      %54 = stablehlo.add %50, %53 : tensor<3x2xf32>
// CHECK-NEXT:      %55 = stablehlo.reshape %54 : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %56 = stablehlo.dynamic_update_slice %iterArg_14, %55, %iterArg, %c_4, %c_4 : (tensor<5x3x2xf32>, tensor<1x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x2xf32>
// CHECK-NEXT:      %57 = stablehlo.tanh %54 : tensor<3x2xf32>
// CHECK-NEXT:      stablehlo.return %40, %iterArg_8, %iterArg_9, %57, %iterArg_11, %39, %37, %56, %45 : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x2xf32>, tensor<3x2x6xf32>, tensor<5x3x3xf32>, tensor<5x3x2xf32>, tensor<5x3x2xf32>, tensor<5x3x2x1xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %13 = stablehlo.abs %12#3 : tensor<3x2xf32>
// CHECK-NEXT:    %14 = stablehlo.transpose %12#4, dims = [2, 1, 0] : (tensor<3x2x6xf32>) -> tensor<6x2x3xf32>
// CHECK-NEXT:    %15 = stablehlo.transpose %12#1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %16 = stablehlo.transpose %12#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %17 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %18 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %arg6, dims = [] : (tensor<f32>) -> tensor<3x2xf32>
// CHECK-NEXT:    %20 = stablehlo.multiply %19, %13 : tensor<3x2xf32>
// CHECK-NEXT:    %21 = stablehlo.add %20, %20 : tensor<3x2xf32>
// CHECK-NEXT:    %22 = stablehlo.compare  GE, %12#3, %cst_7 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xi1>
// CHECK-NEXT:    %23 = stablehlo.negate %21 : tensor<3x2xf32>
// CHECK-NEXT:    %24 = stablehlo.select %22, %21, %23 : tensor<3x2xi1>, tensor<3x2xf32>
// CHECK-NEXT:    %25:7 = stablehlo.while(%iterArg = %c_4, %iterArg_8 = %18, %iterArg_9 = %17, %iterArg_10 = %arg10, %iterArg_11 = %arg11, %iterArg_12 = %24, %iterArg_13 = %c) : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<i64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %36 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %36 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %36 = stablehlo.dynamic_slice %12#5, %iterArg_13, %c_4, %c_4, sizes = [1, 3, 3] : (tensor<5x3x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x3xf32>
// CHECK-NEXT:      %37 = stablehlo.reshape %36 : (tensor<1x3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %38 = stablehlo.dynamic_slice %12#6, %iterArg_13, %c_4, %c_4, sizes = [1, 3, 2] : (tensor<5x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %39 = stablehlo.reshape %38 : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %40 = stablehlo.dynamic_slice %12#7, %iterArg_13, %c_4, %c_4, sizes = [1, 3, 2] : (tensor<5x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %41 = stablehlo.reshape %40 : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %42 = stablehlo.dynamic_slice %12#8, %iterArg_13, %c_4, %c_4, %c_4, sizes = [1, 3, 2, 1] : (tensor<5x3x2x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x2x1xf32>
// CHECK-NEXT:      %43 = stablehlo.reshape %42 : (tensor<1x3x2x1xf32>) -> tensor<3x2x1xf32>
// CHECK-NEXT:      %44 = stablehlo.transpose %43, dims = [2, 1, 0] : (tensor<3x2x1xf32>) -> tensor<1x2x3xf32>
// CHECK-NEXT:      %45 = stablehlo.reshape %44 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:      %46 = stablehlo.add %iterArg, %c_6 {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK-NEXT:      %47 = stablehlo.tanh %41 : tensor<3x2xf32>
// CHECK-NEXT:      %48 = stablehlo.multiply %47, %47 : tensor<3x2xf32>
// CHECK-NEXT:      %49 = stablehlo.subtract %cst_3, %48 : tensor<3x2xf32>
// CHECK-NEXT:      %50 = stablehlo.multiply %iterArg_12, %49 : tensor<3x2xf32>
// CHECK-NEXT:      %51 = stablehlo.dot_general %iterArg_12, %49, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3xf32>
// CHECK-NEXT:      %52 = stablehlo.add %iterArg_10, %51 : tensor<3xf32>
// CHECK-NEXT:      %53 = stablehlo.dot_general %50, %45, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %54 = stablehlo.add %iterArg_8, %53 : tensor<3x3xf32>
// CHECK-NEXT:      %55 = stablehlo.add %iterArg_11, %51 : tensor<3xf32>
// CHECK-NEXT:      %56 = stablehlo.dot_general %50, %39, contracting_dims = [1] x [1] : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %57 = stablehlo.add %iterArg_9, %56 : tensor<3x3xf32>
// CHECK-NEXT:      %58 = stablehlo.dot_general %37, %50, contracting_dims = [0] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %59 = stablehlo.subtract %iterArg_13, %c_6 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %46, %54, %57, %52, %55, %58, %59 : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %26 = stablehlo.multiply %11, %11 : tensor<3x2xf32>
// CHECK-NEXT:    %27 = stablehlo.subtract %cst_3, %26 : tensor<3x2xf32>
// CHECK-NEXT:    %28 = stablehlo.multiply %25#5, %27 : tensor<3x2xf32>
// CHECK-NEXT:    %29 = stablehlo.dot_general %25#5, %27, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3xf32>
// CHECK-NEXT:    %30 = stablehlo.add %25#3, %29 : tensor<3xf32>
// CHECK-NEXT:    %31 = stablehlo.dot_general %28, %5, contracting_dims = [1] x [0] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %32 = stablehlo.add %25#4, %29 : tensor<3xf32>
// CHECK-NEXT:    %33 = stablehlo.transpose %25#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %34 = stablehlo.add %31, %25#1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<3x3xf32>
// CHECK-NEXT:    %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    return %14, %15, %16, %arg3, %arg4, %arg5, %35, %33, %30, %32 : tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:  }
