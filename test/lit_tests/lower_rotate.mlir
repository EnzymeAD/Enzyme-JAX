// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=lower_rotate" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

module @"reactant_loop!" {
  func.func @main(%0: tensor<4x8x80xf64>) -> tensor<4x8x80xf64> {
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    stablehlo.return %1 : tensor<4x8x80xf64>
  }
}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x8x80xf64>) -> tensor<4x8x80xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:4, 0:8, 2:80] : (tensor<4x8x80xf64>) -> tensor<4x8x78xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:4, 0:8, 0:2] : (tensor<4x8x80xf64>) -> tensor<4x8x2xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.concatenate %[[VAL_1]], %[[VAL_2]], dim = 2 : (tensor<4x8x78xf64>, tensor<4x8x2xf64>) -> tensor<4x8x80xf64>
// CHECK:           stablehlo.return %[[VAL_3]] : tensor<4x8x80xf64>
// CHECK:         }

