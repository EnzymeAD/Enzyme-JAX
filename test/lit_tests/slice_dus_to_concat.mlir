// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @slice_dus_to_concat(
    %iterArg_37: tensor<144x1024x1008xf64>,
    %77: tensor<128x1008x1008xf64>,
    %pad_val_scalar: tensor<f32>
  ) -> tensor<128x1008x1008xf64> { 

    %c_30 = stablehlo.constant dense<8> : tensor<i64>

    %c_33 = stablehlo.constant dense<0> : tensor<i64>

      %78 = stablehlo.dynamic_update_slice %iterArg_37, %77, %c_30, %c_30, %c_33 : (tensor<144x1024x1008xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>

      %275 = stablehlo.slice %78 [8:136, 7:1015, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>

    func.return %275 : tensor<128x1008x1008xf64>
  }
}

// CHECK:  func.func @slice_dus_to_concat(%arg0: tensor<144x1024x1008xf64>, %arg1: tensor<128x1008x1008xf64>, %arg2: tensor<f32>) -> tensor<128x1008x1008xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:136, 7:8, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:128, 0:1007, 0:1008] : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<128x1x1008xf64>, tensor<128x1007x1008xf64>) -> tensor<128x1008x1008xf64>
// CHECK-NEXT:    return %2 : tensor<128x1008x1008xf64>
// CHECK-NEXT:  }
