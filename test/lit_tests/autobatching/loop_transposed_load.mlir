// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg1: tensor<f32>, %arg4: tensor<2048x2048xf32>, %prod: tensor<2048x2048x1xf32>) -> tensor<2048x2048xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<2048> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg4) : tensor<i64>, tensor<2048x2048xf32>
    cond {
      %2 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.transpose %iterArg_4, dims = [1, 0] : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %2, %5, %c, sizes = [1, 2048] : (tensor<2048x2048xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2048xf32>
      %7 = stablehlo.reshape %6 : (tensor<1x2048xf32>) -> tensor<2048x1xf32>
      %8 = stablehlo.multiply %7, %0 : tensor<2048x1xf32>
      %9 = stablehlo.dynamic_slice %prod, %iterArg, %c_1, %c_1, sizes = [1, 2048, 1] : (tensor<2048x2048x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2048x1xf32>
      %10 = stablehlo.reshape %9 : (tensor<1x2048x1xf32>) -> tensor<2048x1xf32>
      %11 = stablehlo.add %10, %8 : tensor<2048x1xf32>
      %12 = stablehlo.dynamic_update_slice %iterArg_4, %11, %c, %5 : (tensor<2048x2048xf32>, tensor<2048x1xf32>, tensor<i32>, tensor<i32>) -> tensor<2048x2048xf32>
      stablehlo.return %3, %12 : tensor<i64>, tensor<2048x2048xf32>
    }
    return %1#1 : tensor<2048x2048xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: stablehlo.while
// CHECK: %[[BC:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [1, 0] : (tensor<2048x2048xf32>) -> tensor<2048x2048x1xf32>
// CHECK: %[[BETA:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2048x2048x1xf32>
// CHECK: %[[MUL:.+]] = stablehlo.multiply %[[BC]], %[[BETA]] : tensor<2048x2048x1xf32>
// CHECK: %[[ADD:.+]] = stablehlo.add %arg2, %[[MUL]] : tensor<2048x2048x1xf32>
// CHECK: %[[RS:.+]] = stablehlo.reshape %[[ADD]]
// CHECK: %[[T:.+]] = stablehlo.transpose %[[RS]], dims = [1, 0] : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
// CHECK: return %[[T]]
// CHECK-NOT: stablehlo.while
