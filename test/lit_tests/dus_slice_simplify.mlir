// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dus_slice_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
module {
  func.func @foo(%in: tensor<528x2048x2048xf64>, %update: tensor<1x2032x2032xf64>) -> (tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>) {
    %c_22 = stablehlo.constant dense<2040> : tensor<i32>
    %c_20 = stablehlo.constant dense<520> : tensor<i32>
    %c_23 = stablehlo.constant dense<8> : tensor<i32>
    %dus = stablehlo.dynamic_update_slice %in, %update, %c_20, %c_23, %c_23 : (tensor<528x2048x2048xf64>, tensor<1x2032x2032xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<528x2048x2048xf64>
    %r1 = stablehlo.slice %dus [0:528, 0:2048, 8:2040] : (tensor<528x2048x2048xf64>) -> tensor<528x2048x2032xf64>
    %r2 = stablehlo.slice %dus [9:521, 8:2040, 8:2040] : (tensor<528x2048x2048xf64>) -> tensor<512x2032x2032xf64>
    return %r1, %r2 : tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>
  }
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<528x2048x2048xf64>,
// CHECK-SAME:                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1x2032x2032xf64>) -> (tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:528, 0:2048, 8:2040] : (tensor<528x2048x2048xf64>) -> tensor<528x2048x2032xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.slice %[[VAL_1]] [0:1, 0:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x2032x2032xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.dynamic_update_slice %[[VAL_3]], %[[VAL_4]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (tensor<528x2048x2032xf64>, tensor<1x2032x2032xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<528x2048x2032xf64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.slice %[[VAL_5]] [9:521, 8:2040, 0:2032] : (tensor<528x2048x2032xf64>) -> tensor<512x2032x2032xf64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.slice %[[VAL_5]] [0:528, 0:2048, 0:2032] : (tensor<528x2048x2032xf64>) -> tensor<528x2048x2032xf64>
// CHECK:           return %[[VAL_7]], %[[VAL_6]] : tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>
// CHECK:         }

