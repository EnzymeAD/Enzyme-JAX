// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @dus_pad_operand_test(
    %1162 : tensor<1x6128x12272xf64>,
    %1217 : tensor<1x6128x12272xf64>,
    %pad_val_scalar: tensor<f64>
  ) -> (tensor<1x6144x12288xf64>, tensor<1x6144x12288xf64>) {

%c_308 = stablehlo.constant dense<8> : tensor<i32>
%c_351 = stablehlo.constant dense<0> : tensor<i32>

%1165 = stablehlo.pad %1162, %pad_val_scalar, low = [0, 8, 8], high = [0, 8, 8], interior = [0, 0, 0] : (tensor<1x6128x12272xf64>, tensor<f64>) -> tensor<1x6144x12288xf64>

%1219 = stablehlo.dynamic_update_slice %1165, %1217, %c_351, %c_308, %c_308 : (tensor<1x6144x12288xf64>, tensor<1x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x6144x12288xf64>

    func.return %1219, %1165 : tensor<1x6144x12288xf64>, tensor<1x6144x12288xf64>
  }
}

// CHECK:  func.func @dus_pad_operand_test(%arg0: tensor<1x6128x12272xf64>, %arg1: tensor<1x6128x12272xf64>, %arg2: tensor<f64>) -> (tensor<1x6144x12288xf64>, tensor<1x6144x12288xf64>) {
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %arg2, low = [0, 8, 8], high = [0, 8, 8], interior = [0, 0, 0] : (tensor<1x6128x12272xf64>, tensor<f64>) -> tensor<1x6144x12288xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %arg1, %arg2, low = [0, 8, 8], high = [0, 8, 8], interior = [0, 0, 0] : (tensor<1x6128x12272xf64>, tensor<f64>) -> tensor<1x6144x12288xf64>
// CHECK-NEXT:    return %1, %0 : tensor<1x6144x12288xf64>, tensor<1x6144x12288xf64>
// CHECK-NEXT:  }
