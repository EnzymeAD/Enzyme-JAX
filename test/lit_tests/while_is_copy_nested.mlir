// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<4x4xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c : tensor<i32>
      %5:2 = stablehlo.while(%iterArg_4 = %c_0, %iterArg_5 = %iterArg_3) : tensor<i64>, tensor<4x4xf32>
      cond {
        %6 = stablehlo.compare  LT, %iterArg_4, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %6 : tensor<i1>
      } do {
        %6 = stablehlo.add %c_2, %iterArg_4 : tensor<i64>
        %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
        %8 = stablehlo.subtract %7, %c : tensor<i32>
        %9 = stablehlo.dynamic_slice %0, %4, %8, sizes = [1, 1] : (tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %10 = stablehlo.dynamic_update_slice %iterArg_5, %9, %4, %8 : (tensor<4x4xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
        stablehlo.return %6, %10 : tensor<i64>, tensor<4x4xf32>
      }
      stablehlo.return %2, %5#1 : tensor<i64>, tensor<4x4xf32>
    }
    return %1#1 : tensor<4x4xf32>
}

// CHECK: func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     return %0 : tensor<4x4xf32>
// CHECK-NEXT: }
