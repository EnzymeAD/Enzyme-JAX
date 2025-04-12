// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{rotate_comm=1 rotate_to_pad_comm=0})" %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=4, "y"=1, "z"=1]>

// CHECK-LABEL:   func.func @left_to_right(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x8x80xf64>) -> tensor<4x8x80xf64> {
// CHECK:           %[[VAL_1:.*]] = sdy.manual_computation(%[[VAL_0]]) in_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] out_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] manual_axes={"x", "y", "z"} (%[[VAL_2:.*]]: tensor<4x8x20xf64>) {
// CHECK:             %[[VAL_3:.*]] = stablehlo.slice %[[VAL_2]] [0:4, 0:8, 0:1] : (tensor<4x8x20xf64>) -> tensor<4x8x1xf64>
// CHECK:             %[[VAL_4:.*]] = "stablehlo.collective_permute"(%[[VAL_3]]) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, source_target_pairs = dense<{{\[\[}}1, 0], [2, 1], [3, 2], [0, 3]]> : tensor<4x2xi64>}> : (tensor<4x8x1xf64>) -> tensor<4x8x1xf64>
// CHECK:             %[[VAL_5:.*]] = stablehlo.slice %[[VAL_2]] [0:4, 0:8, 1:20] : (tensor<4x8x20xf64>) -> tensor<4x8x19xf64>
// CHECK:             %[[VAL_6:.*]] = stablehlo.concatenate %[[VAL_5]], %[[VAL_4]], dim = 2 : (tensor<4x8x19xf64>, tensor<4x8x1xf64>) -> tensor<4x8x20xf64>
// CHECK:             sdy.return %[[VAL_6]] : tensor<4x8x20xf64>
// CHECK:           } : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK:           return %[[VAL_1]] : tensor<4x8x80xf64>
// CHECK:         }

  func.func @left_to_right(%arg: tensor<4x8x80xf64>) -> (tensor<4x8x80xf64>) {
    %res = "enzymexla.rotate"(%arg) <{amount = 1 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    func.return %res : tensor<4x8x80xf64>
  }

// CHECK-LABEL:   func.func @right_to_left(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x8x80xf64>) -> tensor<4x8x80xf64> {
// CHECK:           %[[VAL_1:.*]] = sdy.manual_computation(%[[VAL_0]]) in_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] out_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] manual_axes={"x", "y", "z"} (%[[VAL_2:.*]]: tensor<4x8x20xf64>) {
// CHECK:             %[[VAL_3:.*]] = stablehlo.slice %[[VAL_2]] [0:4, 0:8, 19:20] : (tensor<4x8x20xf64>) -> tensor<4x8x1xf64>
// CHECK:             %[[VAL_4:.*]] = "stablehlo.collective_permute"(%[[VAL_3]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<{{\[\[}}3, 0], [0, 1], [1, 2], [2, 3]]> : tensor<4x2xi64>}> : (tensor<4x8x1xf64>) -> tensor<4x8x1xf64>
// CHECK:             %[[VAL_5:.*]] = stablehlo.slice %[[VAL_2]] [0:4, 0:8, 0:19] : (tensor<4x8x20xf64>) -> tensor<4x8x19xf64>
// CHECK:             %[[VAL_6:.*]] = stablehlo.concatenate %[[VAL_4]], %[[VAL_5]], dim = 2 : (tensor<4x8x1xf64>, tensor<4x8x19xf64>) -> tensor<4x8x20xf64>
// CHECK:             sdy.return %[[VAL_6]] : tensor<4x8x20xf64>
// CHECK:           } : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK:           return %[[VAL_1]] : tensor<4x8x80xf64>
// CHECK:         }

  func.func @right_to_left(%arg: tensor<4x8x80xf64>) -> (tensor<4x8x80xf64>) {
    %res = "enzymexla.rotate"(%arg) <{amount = 79 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    func.return %res : tensor<4x8x80xf64>
  }
}
