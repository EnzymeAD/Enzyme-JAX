// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%a : tensor<3x4xf32>, %b : tensor<3x4xf32>) -> tensor<2x3x4xf32> {
    %u = stablehlo.reshape %a : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %v = stablehlo.reshape %b : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %concat = stablehlo.concatenate %u, %v, dim=0 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<2x3x4xf32>
    return %concat : tensor<2x3x4xf32>
  }

  func.func @main2(%a : tensor<3x4xf32>, %b : tensor<3x4xf32>) -> tensor<2x3x4xf64> {
    %u = stablehlo.reshape %a : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %uc = stablehlo.convert %u : (tensor<1x3x4xf32>) -> tensor<1x3x4xf64>
    %v = stablehlo.reshape %b : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %vc = stablehlo.convert %v : (tensor<1x3x4xf32>) -> tensor<1x3x4xf64>
    %concat = stablehlo.concatenate %uc, %vc, dim=0 : (tensor<1x3x4xf64>, tensor<1x3x4xf64>) -> tensor<2x3x4xf64>
    return %concat : tensor<2x3x4xf64>
  }

  // TODO this opt
  func.func @main3(%a : tensor<3x4xf32>, %b : tensor<3x4xf32>) -> tensor<3x2x4xf32> {
    %u = stablehlo.reshape %a : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    %v = stablehlo.reshape %b : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    %concat = stablehlo.concatenate %u, %v, dim=1 : (tensor<3x1x4xf32>, tensor<3x1x4xf32>) -> tensor<3x2x4xf32>
    return %concat : tensor<3x2x4xf32>
  }

  func.func @main4(%a : tensor<f32>, %b : tensor<f32>) -> tensor<2xf32> {
    %u = stablehlo.reshape %a : (tensor<f32>) -> tensor<1xf32>
    %v = stablehlo.reshape %b : (tensor<f32>) -> tensor<1xf32>
    %concat = stablehlo.concatenate %u, %v, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    return %concat : tensor<2xf32>
  }
}


// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<3x4xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: tensor<3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.concatenate %[[VAL_0]], %[[VAL_1]], dim = 0 : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<6x4xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<6x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           return %[[VAL_3]] : tensor<2x3x4xf32>
// CHECK:         }

// CHECK-LABEL:   func.func @main2(
// CHECK-SAME:                     %[[VAL_0:.*]]: tensor<3x4xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: tensor<3x4xf32>) -> tensor<2x3x4xf64> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.convert %[[VAL_0]] : (tensor<3x4xf32>) -> tensor<3x4xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.concatenate %[[VAL_2]], %[[VAL_3]], dim = 0 : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<6x4xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.reshape %[[VAL_4]] : (tensor<6x4xf64>) -> tensor<2x3x4xf64>
// CHECK:           return %[[VAL_5]] : tensor<2x3x4xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @main4(
// CHECK-SAME:                     %[[VAL_0:.*]]: tensor<f32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: tensor<f32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.concatenate %[[VAL_2]], %[[VAL_3]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// CHECK:           return %[[VAL_4]] : tensor<2xf32>
// CHECK:         }

