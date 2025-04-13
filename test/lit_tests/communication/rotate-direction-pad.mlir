// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{rotate_comm=1 rotate_to_pad_comm=0})" %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=4, "y"=5, "z"=1]>

  func.func @left_to_right(%arg: tensor<4x8x80xf64>) -> (tensor<4x8x80xf64>) {
    %res = "enzymexla.rotate"(%arg) <{amount = 1 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    func.return %res : tensor<4x8x80xf64>
  }

  func.func @right_to_left(%arg: tensor<4x8x80xf64>) -> (tensor<4x8x80xf64>) {
    %res = "enzymexla.rotate"(%arg) <{amount = 79 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    func.return %res : tensor<4x8x80xf64>
  }
}

// CHECK:  func.func @left_to_right(%arg0: tensor<4x8x80xf64>) -> tensor<4x8x80xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<4x8x80xf64>, tensor<f64>) -> tensor<4x10x80xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] out_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] manual_axes={"x", "y", "z"} (%arg1: tensor<4x2x20xf64>) {
// CHECK-NEXT:      %3 = stablehlo.slice %arg1 [0:4, 0:2, 0:1] : (tensor<4x2x20xf64>) -> tensor<4x2x1xf64>
// CHECK-NEXT{LITERAL}:      %4 = "stablehlo.collective_permute"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, source_target_pairs = dense<[[5, 0], [6, 1], [7, 2], [8, 3], [9, 4], [10, 5], [11, 6], [12, 7], [13, 8], [14, 9], [15, 10], [16, 11], [17, 12], [18, 13], [19, 14], [0, 15], [1, 16], [2, 17], [3, 18], [4, 19]]> : tensor<20x2xi64>}> : (tensor<4x2x1xf64>) -> tensor<4x2x1xf64>
// CHECK-NEXT:      %5 = stablehlo.slice %arg1 [0:4, 0:2, 1:20] : (tensor<4x2x20xf64>) -> tensor<4x2x19xf64>
// CHECK-NEXT:      %6 = stablehlo.concatenate %5, %4, dim = 2 : (tensor<4x2x19xf64>, tensor<4x2x1xf64>) -> tensor<4x2x20xf64>
// CHECK-NEXT:      sdy.return %6 : tensor<4x2x20xf64>
// CHECK-NEXT:    } : (tensor<4x10x80xf64>) -> tensor<4x10x80xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:4, 0:8, 0:80] : (tensor<4x10x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    return %2 : tensor<4x8x80xf64>
// CHECK-NEXT:  }
// CHECK:  func.func @right_to_left(%arg0: tensor<4x8x80xf64>) -> tensor<4x8x80xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<4x8x80xf64>, tensor<f64>) -> tensor<4x10x80xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] out_shardings=[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>] manual_axes={"x", "y", "z"} (%arg1: tensor<4x2x20xf64>) {
// CHECK-NEXT:      %3 = stablehlo.slice %arg1 [0:4, 0:2, 19:20] : (tensor<4x2x20xf64>) -> tensor<4x2x1xf64>
// CHECK-NEXT{LITERAL}:      %4 = "stablehlo.collective_permute"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[15, 0], [16, 1], [17, 2], [18, 3], [19, 4], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [7, 12], [8, 13], [9, 14], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19]]> : tensor<20x2xi64>}> : (tensor<4x2x1xf64>) -> tensor<4x2x1xf64>
// CHECK-NEXT:      %5 = stablehlo.slice %arg1 [0:4, 0:2, 0:19] : (tensor<4x2x20xf64>) -> tensor<4x2x19xf64>
// CHECK-NEXT:      %6 = stablehlo.concatenate %4, %5, dim = 2 : (tensor<4x2x1xf64>, tensor<4x2x19xf64>) -> tensor<4x2x20xf64>
// CHECK-NEXT:      sdy.return %6 : tensor<4x2x20xf64>
// CHECK-NEXT:    } : (tensor<4x10x80xf64>) -> tensor<4x10x80xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:4, 0:8, 0:80] : (tensor<4x10x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    return %2 : tensor<4x8x80xf64>
// CHECK-NEXT:  }
