// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="enable_auto_batching_passes=true" | FileCheck %s

module {
  func.func @main(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>, %arg2: tensor<3x2xf64>) -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x2xf64>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<3> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<3x2xf64> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c : tensor<i32>
      %6:2 = stablehlo.while(%iterArg_5 = %c_1, %iterArg_6 = %iterArg_4) : tensor<i64>, tensor<3x2xf64> attributes {enzyme.disable_mincut}
      cond {
        %7 = stablehlo.compare  LT, %iterArg_5, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %7 : tensor<i1>
      } do {
        %7 = stablehlo.add %c_3, %iterArg_5 : tensor<i64>
        %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
        %9 = stablehlo.subtract %8, %c : tensor<i32>
        %10 = stablehlo.dynamic_slice %0, %5, %9, sizes = [1, 1] : (tensor<2x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
        %11 = stablehlo.dynamic_slice %1, %5, %9, sizes = [1, 1] : (tensor<2x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
        %12 = stablehlo.add %10, %11 : tensor<1x1xf64>
        %13 = stablehlo.dynamic_update_slice %iterArg_6, %12, %9, %5 : (tensor<3x2xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<3x2xf64>
        stablehlo.return %7, %13 : tensor<i64>, tensor<3x2xf64>
      }
      stablehlo.return %3, %6#1 : tensor<i64>, tensor<3x2xf64>
    }
    return %2#0, %c_3, %c_3, %c_2, %c_0, %2#1 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x2xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>, %arg2: tensor<3x2xf64>) -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x2xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.add %arg1, %arg2 : tensor<3x2xf64>
// CHECK-NEXT:    return %c_0, %c_1, %c_1, %c_0, %c, %0 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<3x2xf64>
// CHECK-NEXT:  }
