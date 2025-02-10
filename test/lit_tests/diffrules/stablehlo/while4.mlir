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
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<5x2x3xf32>
// CHECK-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<5x3x2xf32>
// CHECK-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<5x3x3xf32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<3x2xf32>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf32>
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
// CHECK-NEXT:    %12:12 = stablehlo.while(%iterArg = %c_4, %iterArg_9 = %1, %iterArg_10 = %2, %iterArg_11 = %arg3, %iterArg_12 = %arg4, %iterArg_13 = %11, %iterArg_14 = %0, %iterArg_15 = %c_4, %iterArg_16 = %cst_1, %iterArg_17 = %cst_0, %iterArg_18 = %cst, %iterArg_19 = %cst_0) : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<3x2x6xf32>, tensor<i64>, tensor<5x3x3xf32>, tensor<5x3x2xf32>, tensor<5x2x3xf32>, tensor<5x3x2xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %37 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %37 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %37 = stablehlo.add %iterArg_15, %c_6 : tensor<i64>
// CHECK-NEXT:      %38 = stablehlo.add %c_5, %iterArg : tensor<i64>
// CHECK-NEXT:      %39 = stablehlo.subtract %38, %c_6 : tensor<i64>
// CHECK-NEXT:      %40 = stablehlo.dynamic_slice %iterArg_14, %c_4, %c_4, %39, sizes = [3, 2, 1] : (tensor<3x2x6xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x2x1xf32>
// CHECK-NEXT:      %41 = stablehlo.transpose %40, dims = [2, 1, 0] : (tensor<3x2x1xf32>) -> tensor<1x2x3xf32>
// CHECK-NEXT:      %42 = stablehlo.reshape %41 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:      %43 = stablehlo.reshape %iterArg_10 : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
// CHECK-NEXT:      %44 = stablehlo.dynamic_update_slice %iterArg_16, %43, %iterArg, %c_4, %c_4 : (tensor<5x3x3xf32>, tensor<1x3x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x3xf32>
// CHECK-NEXT:      %45 = stablehlo.reshape %iterArg_13 : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %46 = stablehlo.dynamic_update_slice %iterArg_17, %45, %iterArg, %c_4, %c_4 : (tensor<5x3x2xf32>, tensor<1x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x2xf32>
// CHECK-NEXT:      %47 = stablehlo.dot_general %iterArg_10, %iterArg_13, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %48 = stablehlo.broadcast_in_dim %iterArg_12, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %49 = stablehlo.add %47, %48 : tensor<3x2xf32>
// CHECK-NEXT:      %50 = stablehlo.reshape %42 : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
// CHECK-NEXT:      %51 = stablehlo.dynamic_update_slice %iterArg_18, %50, %iterArg, %c_4, %c_4 : (tensor<5x2x3xf32>, tensor<1x2x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x2x3xf32>
// CHECK-NEXT:      %52 = stablehlo.dot_general %iterArg_9, %42, contracting_dims = [1] x [1] : (tensor<3x3xf32>, tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %53 = stablehlo.broadcast_in_dim %iterArg_11, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %54 = stablehlo.add %52, %53 : tensor<3x2xf32>
// CHECK-NEXT:      %55 = stablehlo.add %49, %54 : tensor<3x2xf32>
// CHECK-NEXT:      %56 = stablehlo.reshape %55 : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %57 = stablehlo.dynamic_update_slice %iterArg_19, %56, %iterArg, %c_4, %c_4 : (tensor<5x3x2xf32>, tensor<1x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<5x3x2xf32>
// CHECK-NEXT:      %58 = stablehlo.tanh %55 : tensor<3x2xf32>
// CHECK-NEXT:      %59 = stablehlo.add %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %59, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %58, %iterArg_14, %37, %44, %46, %51, %57 : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<3x2x6xf32>, tensor<i64>, tensor<5x3x3xf32>, tensor<5x3x2xf32>, tensor<5x2x3xf32>, tensor<5x3x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %13 = stablehlo.abs %12#5 : tensor<3x2xf32>
// CHECK-NEXT:    %14 = stablehlo.transpose %12#6, dims = [2, 1, 0] : (tensor<3x2x6xf32>) -> tensor<6x2x3xf32>
// CHECK-NEXT:    %15 = stablehlo.transpose %12#1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %16 = stablehlo.transpose %12#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %17 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %18 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %arg6, dims = [] : (tensor<f32>) -> tensor<3x2xf32>
// CHECK-NEXT:    %20 = stablehlo.multiply %19, %13 : tensor<3x2xf32>
// CHECK-NEXT:    %21 = stablehlo.add %20, %20 : tensor<3x2xf32>
// CHECK-NEXT:    %22 = stablehlo.compare  GE, %12#5, %cst_8 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xi1>
// CHECK-NEXT:    %23 = stablehlo.negate %21 : tensor<3x2xf32>
// CHECK-NEXT:    %24 = stablehlo.select %22, %21, %23 : tensor<3x2xi1>, tensor<3x2xf32>
// CHECK-NEXT:    %25:7 = stablehlo.while(%iterArg = %c_4, %iterArg_9 = %18, %iterArg_10 = %17, %iterArg_11 = %arg10, %iterArg_12 = %arg11, %iterArg_13 = %24, %iterArg_14 = %c) : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %37 = stablehlo.compare  LT, %iterArg, %12#7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %37 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %37 = stablehlo.add %iterArg, %c_6 : tensor<i64>
// CHECK-NEXT:      %38 = stablehlo.dynamic_slice %12#11, %iterArg_14, %c_4, %c_4, sizes = [1, 3, 2] : (tensor<5x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %39 = stablehlo.reshape %38 : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %40 = stablehlo.tanh %39 : tensor<3x2xf32>
// CHECK-NEXT:      %41 = stablehlo.multiply %40, %40 : tensor<3x2xf32>
// CHECK-NEXT:      %42 = stablehlo.subtract %cst_3, %41 : tensor<3x2xf32>
// CHECK-NEXT:      %43 = stablehlo.multiply %iterArg_13, %42 : tensor<3x2xf32>
// CHECK-NEXT:      %44 = stablehlo.reduce(%43 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:      %45 = stablehlo.add %iterArg_11, %44 : tensor<3xf32>
// CHECK-NEXT:      %46 = stablehlo.dynamic_slice %12#10, %iterArg_14, %c_4, %c_4, sizes = [1, 2, 3] : (tensor<5x2x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x3xf32>
// CHECK-NEXT:      %47 = stablehlo.reshape %46 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:      %48 = stablehlo.dot_general %43, %47, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %49 = stablehlo.add %iterArg_9, %48 : tensor<3x3xf32>
// CHECK-NEXT:      %50 = stablehlo.add %iterArg_12, %44 : tensor<3xf32>
// CHECK-NEXT:      %51 = stablehlo.dynamic_slice %12#8, %iterArg_14, %c_4, %c_4, sizes = [1, 3, 3] : (tensor<5x3x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x3xf32>
// CHECK-NEXT:      %52 = stablehlo.reshape %51 : (tensor<1x3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %53 = stablehlo.dynamic_slice %12#9, %iterArg_14, %c_4, %c_4, sizes = [1, 3, 2] : (tensor<5x3x2xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x2xf32>
// CHECK-NEXT:      %54 = stablehlo.reshape %53 : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %55 = stablehlo.dot_general %43, %54, contracting_dims = [1] x [1] : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %56 = stablehlo.add %iterArg_10, %55 : tensor<3x3xf32>
// CHECK-NEXT:      %57 = stablehlo.dot_general %52, %43, contracting_dims = [0] x [0] : (tensor<3x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:      %58 = stablehlo.subtract %iterArg_14, %c_6 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %37, %49, %56, %45, %50, %57, %58 : tensor<i64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %26 = stablehlo.multiply %11, %11 : tensor<3x2xf32>
// CHECK-NEXT:    %27 = stablehlo.subtract %cst_3, %26 : tensor<3x2xf32>
// CHECK-NEXT:    %28 = stablehlo.multiply %25#5, %27 : tensor<3x2xf32>
// CHECK-NEXT:    %29 = stablehlo.reduce(%28 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:    %30 = stablehlo.add %25#3, %29 : tensor<3xf32>
// CHECK-NEXT:    %31 = stablehlo.dot_general %28, %5, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %32 = stablehlo.reduce(%28 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:    %33 = stablehlo.add %25#4, %32 : tensor<3xf32>
// CHECK-NEXT:    %34 = stablehlo.transpose %25#2, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    %35 = stablehlo.add %31, %25#1 : tensor<3x3xf32>
// CHECK-NEXT:    %36 = stablehlo.transpose %35, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:    return %14, %15, %16, %12#3, %12#4, %arg5, %36, %34, %30, %33 : tensor<6x2x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2xui64>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:  }
