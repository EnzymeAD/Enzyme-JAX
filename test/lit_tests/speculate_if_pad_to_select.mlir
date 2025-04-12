// RUN: enzymexlamlir-opt %s  --enzyme-hlo-generate-td="patterns=speculate_if_pad_to_select;if_to_select" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s

module {
  func.func @if_pad_to_select(%56: tensor<i1>, %38: tensor<5x46x94xf64>, %55: tensor<4x46x94xf64>, %cst_280 : tensor<f64>) -> tensor<5x46x94xf64> {
    %result = "stablehlo.if"(%56) ({
      stablehlo.return %38 : tensor<5x46x94xf64>
    }, {
      %342 = stablehlo.pad %55, %cst_280, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x46x94xf64>, tensor<f64>) -> tensor<5x46x94xf64>
      stablehlo.return %342 : tensor<5x46x94xf64>
    }) : (tensor<i1>) -> tensor<5x46x94xf64>
    return %result :  tensor<5x46x94xf64>
  }
}

// CHECK:  func.func @if_pad_to_select(%arg0: tensor<i1>, %arg1: tensor<5x46x94xf64>, %arg2: tensor<4x46x94xf64>, %arg3: tensor<f64>) -> tensor<5x46x94xf64> {
// CHECK-NEXT:    %0 = stablehlo.pad %arg2, %arg3, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x46x94xf64>, tensor<f64>) -> tensor<5x46x94xf64>
// CHECK-NEXT:    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<5x46x94xf64>
// CHECK-NEXT:    return %1 : tensor<5x46x94xf64>
// CHECK-NEXT:  }
