// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dus_to_i32" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<144x1024x1008xf64>, %arg1: tensor<128x1x1008xf64>, %arg2: tensor<128x1x1008xf64>) -> tensor<144x1024x1008xf64> {
    %c = stablehlo.constant dense<2147483648> : tensor<i64>
    %c_0 = stablehlo.constant dense<1016> : tensor<i64>
    %c_1 = stablehlo.constant dense<8> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_1, %c, %c_2 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
    %1 = stablehlo.dynamic_update_slice %0, %arg2, %c_1, %c_0, %c_2 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
    return %1 : tensor<144x1024x1008xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<144x1024x1008xf64>, %arg1: tensor<128x1x1008xf64>, %arg2: tensor<128x1x1008xf64>) -> tensor<144x1024x1008xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1016> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<2147483648> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_3, %c_2, %c_4 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %0, %arg2, %c_1, %c_0, %c : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x1024x1008xf64>
// CHECK-NEXT:    return %1 : tensor<144x1024x1008xf64>
// CHECK-NEXT:  }
