// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

func.func @main1(%arg0: tensor<25xf32>) -> tensor<13xf32> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<1xf32>
  %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<5> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<0> : tensor<i64>
  %c_4 = stablehlo.constant dense<10> : tensor<i64>
  %c_5 = stablehlo.constant dense<1> : tensor<i64>
  %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<13xf32>
  %0:2 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %cst_6) : tensor<i64>, tensor<13xf32> attributes {enzyme.disable_mincut}
  cond {
    %1 = stablehlo.compare  LT, %iterArg, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %c_5, %iterArg : tensor<i64>
    %2 = stablehlo.multiply %c_2, %1 : tensor<i64>
    %3 = stablehlo.add %2, %c_1 : tensor<i64>
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.subtract %4, %c : tensor<i32>
    %6 = stablehlo.dynamic_slice %arg0, %5, sizes = [1] : (tensor<25xf32>, tensor<i32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %8 = stablehlo.subtract %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.sine %8 : tensor<1xf32>
    %10 = stablehlo.add %1, %c_2 : tensor<i64>
    %11 = stablehlo.convert %10 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.subtract %11, %c : tensor<i32>
    %13 = stablehlo.dynamic_update_slice %iterArg_7, %9, %12 : (tensor<13xf32>, tensor<1xf32>, tensor<i32>) -> tensor<13xf32>
    stablehlo.return %1, %13 : tensor<i64>, tensor<13xf32>
  }
  return %0#1 : tensor<13xf32>
}

// CHECK: func.func @main1(%arg0: tensor<25xf32>) -> tensor<13xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<10xf32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [6:25:2] : (tensor<25xf32>) -> tensor<10xf32>
// CHECK-NEXT:   %1 = stablehlo.multiply %0, %cst_1 : tensor<10xf32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %cst_0 : tensor<10xf32>
// CHECK-NEXT:   %3 = stablehlo.sine %2 : tensor<10xf32>
// CHECK-NEXT:   %4 = stablehlo.pad %3, %cst, low = [2], high = [1], interior = [0] : (tensor<10xf32>, tensor<f32>) -> tensor<13xf32>
// CHECK-NEXT:   return %4 : tensor<13xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<25xf32>) -> tensor<13xf32> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<1xf32>
  %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<5> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<0> : tensor<i64>
  %c_4 = stablehlo.constant dense<10> : tensor<i64>
  %c_5 = stablehlo.constant dense<1> : tensor<i64>
  %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<13xf32>
  %0:10 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %cst_6, %iterArg_8 = %c_4, %iterArg_9 = %c_5, %iterArg_10 = %c_2, %iterArg_11 = %c_1, %iterArg_12 = %c, %iterArg_13 = %arg0, %iterArg_14 = %cst, %iterArg_15 = %cst_0) : tensor<i64>, tensor<13xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i32>, tensor<25xf32>, tensor<1xf32>, tensor<1xf32>
  cond {
    %1 = stablehlo.compare  LT, %iterArg, %iterArg_8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg_9, %iterArg : tensor<i64>
    %2 = stablehlo.multiply %iterArg_10, %1 : tensor<i64>
    %3 = stablehlo.add %2, %iterArg_11 : tensor<i64>
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.subtract %4, %iterArg_12 : tensor<i32>
    %6 = stablehlo.dynamic_slice %iterArg_13, %5, sizes = [1] : (tensor<25xf32>, tensor<i32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %iterArg_14 : tensor<1xf32>
    %8 = stablehlo.subtract %7, %iterArg_15 : tensor<1xf32>
    %9 = stablehlo.sine %8 : tensor<1xf32>
    %10 = stablehlo.add %1, %iterArg_10 : tensor<i64>
    %11 = stablehlo.convert %10 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.subtract %11, %iterArg_12 : tensor<i32>
    %13 = stablehlo.dynamic_update_slice %iterArg_7, %9, %12 : (tensor<13xf32>, tensor<1xf32>, tensor<i32>) -> tensor<13xf32>
    stablehlo.return %1, %13, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15 : tensor<i64>, tensor<13xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i32>, tensor<25xf32>, tensor<1xf32>, tensor<1xf32>
  }
  return %0#1 : tensor<13xf32>
}

// CHECK: func.func @main2(%arg0: tensor<25xf32>) -> tensor<13xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<10xf32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [6:25:2] : (tensor<25xf32>) -> tensor<10xf32>
// CHECK-NEXT:   %1 = stablehlo.multiply %0, %cst_1 : tensor<10xf32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %cst_0 : tensor<10xf32>
// CHECK-NEXT:   %3 = stablehlo.sine %2 : tensor<10xf32>
// CHECK-NEXT:   %4 = stablehlo.pad %3, %cst, low = [2], high = [1], interior = [0] : (tensor<10xf32>, tensor<f32>) -> tensor<13xf32>
// CHECK-NEXT:   return %4 : tensor<13xf32>
// CHECK-NEXT: }
