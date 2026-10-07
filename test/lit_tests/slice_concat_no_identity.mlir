// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_concat" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @whole(%a: tensor<8x14xf64>, %b: tensor<8x1xf64>, %c: tensor<8x4xf64>) -> tensor<8x19xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<8x14xf64>, tensor<8x1xf64>, tensor<8x4xf64>) -> tensor<8x19xf64>
  %1 = stablehlo.slice %0 [0:8, 0:19] : (tensor<8x19xf64>) -> tensor<8x19xf64>
  return %1 : tensor<8x19xf64>
}

// CHECK:       func.func @whole(%[[ARG0:[a-z0-9_]+]]: tensor<8x14xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<8x1xf64>, %[[ARG2:[a-z0-9_]+]]: tensor<8x4xf64>) -> tensor<8x19xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], %[[ARG2]], dim = 1 : (tensor<8x14xf64>, tensor<8x1xf64>, tensor<8x4xf64>) -> tensor<8x19xf64>
// CHECK-NEXT:  return %[[V0]] : tensor<8x19xf64>
// CHECK-NEXT:  }

func.func @wholepieces(%a: tensor<8x14xf64>, %b: tensor<8x1xf64>, %c: tensor<8x4xf64>) -> tensor<8x17xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<8x14xf64>, tensor<8x1xf64>, tensor<8x4xf64>) -> tensor<8x19xf64>
  %1 = stablehlo.slice %0 [0:8, 0:17] : (tensor<8x19xf64>) -> tensor<8x17xf64>
  return %1 : tensor<8x17xf64>
}

// CHECK:       func.func @wholepieces(%[[ARG0:[a-z0-9_]+]]: tensor<8x14xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<8x1xf64>, %[[ARG2:[a-z0-9_]+]]: tensor<8x4xf64>) -> tensor<8x17xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.slice %[[ARG2]] [0:8, 0:2] : (tensor<8x4xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:  %[[V1:[a-z0-9_]+]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], %[[V0]], dim = 1 : (tensor<8x14xf64>, tensor<8x1xf64>, tensor<8x2xf64>) -> tensor<8x17xf64>
// CHECK-NEXT:  return %[[V1]] : tensor<8x17xf64>
// CHECK-NEXT:  }

func.func @onepiece(%a: tensor<8x14xf64>, %b: tensor<8x1xf64>, %c: tensor<8x4xf64>) -> tensor<8x3xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, dim = 1 : (tensor<8x14xf64>, tensor<8x1xf64>, tensor<8x4xf64>) -> tensor<8x19xf64>
  %1 = stablehlo.slice %0 [0:8, 2:5] : (tensor<8x19xf64>) -> tensor<8x3xf64>
  return %1 : tensor<8x3xf64>
}

// CHECK:       func.func @onepiece(%[[ARG0:[a-z0-9_]+]]: tensor<8x14xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<8x1xf64>, %[[ARG2:[a-z0-9_]+]]: tensor<8x4xf64>) -> tensor<8x3xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.slice %[[ARG0]] [0:8, 2:5] : (tensor<8x14xf64>) -> tensor<8x3xf64>
// CHECK-NEXT:  return %[[V0]] : tensor<8x3xf64>
// CHECK-NEXT:  }
