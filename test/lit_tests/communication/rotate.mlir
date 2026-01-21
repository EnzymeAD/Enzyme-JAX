// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{rotate_comm=1 rotate_to_pad_comm=0 rotate_spmd=0})" %s | FileCheck %s --check-prefix=CPERM
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{rotate_comm=0 rotate_to_pad_comm=1 rotate_spmd=0})" %s | FileCheck %s --check-prefix=PAD
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{rotate_comm=0 rotate_to_pad_comm=0 rotate_spmd=1})" %s | FileCheck %s --check-prefix=SPMD

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    return %1 : tensor<4x8x80xf64>
}

func.func @main3(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 1 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    return %1 : tensor<4x8x80xf64>
}

// CPERM: module {
// CPERM-NEXT:   sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// CPERM-NEXT:   func.func @main1(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CPERM-NEXT:     %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// CPERM-NEXT:     %1 = sdy.manual_computation(%0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<4x2x20xf64>) {
// CPERM-NEXT:       %2 = stablehlo.slice %arg1 [0:4, 0:2, 0:2] : (tensor<4x2x20xf64>) -> tensor<4x2x2xf64>
// CPERM-NEXT{LITERAL}:       %3 = "stablehlo.collective_permute"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7], [12, 8], [13, 9], [14, 10], [15, 11], [0, 12], [1, 13], [2, 14], [3, 15]]> : tensor<16x2xi64>}> : (tensor<4x2x2xf64>) -> tensor<4x2x2xf64>
// CPERM-NEXT:       %4 = stablehlo.slice %arg1 [0:4, 0:2, 2:20] : (tensor<4x2x20xf64>) -> tensor<4x2x18xf64>
// CPERM-NEXT:       %5 = stablehlo.concatenate %4, %3, dim = 2 : (tensor<4x2x18xf64>, tensor<4x2x2xf64>) -> tensor<4x2x20xf64>
// CPERM-NEXT:       sdy.return %5 : tensor<4x2x20xf64>
// CPERM-NEXT:     } : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CPERM-NEXT:     return %1 : tensor<4x8x80xf64>
// CPERM-NEXT:   }
// CPERM-NEXT:   func.func @main3(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CPERM-NEXT:      %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// CPERM-NEXT:      %1 = sdy.manual_computation(%0) in_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh1, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "x", "y"} (%arg1: tensor<4x2x20xf64>) {
// CPERM-NEXT:         %2 = stablehlo.slice %arg1 [0:4, 0:2, 0:20] : (tensor<4x2x20xf64>) -> tensor<4x2x20xf64>
// CPERM-NEXT{LITERAL}:         %3 = "stablehlo.collective_permute"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[1, 0], [2, 1], [3, 2], [0, 3], [5, 4], [6, 5], [7, 6], [4, 7], [9, 8], [10, 9], [11, 10], [8, 11], [13, 12], [14, 13], [15, 14], [12, 15]]> : tensor<16x2xi64>}> : (tensor<4x2x20xf64>) -> tensor<4x2x20xf64>
// CPERM-NEXT:       sdy.return %3 : tensor<4x2x20xf64>
// CPERM-NEXT:     } : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CPERM-NEXT:      return %1 : tensor<4x8x80xf64>
// CPERM-NEXT:    }
// CPERM-NEXT:  }

// PAD: module {
// PAD-NEXT:   sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
// PAD-NEXT:   func.func @main1(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// PAD-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:8, 2:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x78xf64>
// PAD-NEXT:     %2 = stablehlo.slice %0 [0:4, 0:8, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x2xf64>
// PAD-NEXT:     %3 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [0, 0, 2], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x78xf64>, tensor<f64>) -> tensor<4x8x80xf64>
// PAD-NEXT:     %4 = stablehlo.pad %2, %cst, low = [0, 0, 78], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x8x2xf64>, tensor<f64>) -> tensor<4x8x80xf64>
// PAD-NEXT:     %5 = stablehlo.add %3, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<4x8x80xf64>
// PAD-NEXT:     return %5 : tensor<4x8x80xf64>
// PAD-NEXT:   }
// PAD-NEXT:   func.func @main3(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// PAD-NEXT:     %1 = stablehlo.slice %0 [0:4, 2:8, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x6x80xf64>
// PAD-NEXT:     %2 = stablehlo.slice %0 [0:4, 0:2, 0:80] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x2x80xf64>
// PAD-NEXT:     %3 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6x80xf64>, tensor<f64>) -> tensor<4x8x80xf64>
// PAD-NEXT:     %4 = stablehlo.pad %2, %cst, low = [0, 6, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x2x80xf64>, tensor<f64>) -> tensor<4x8x80xf64>
// PAD-NEXT:     %5 = stablehlo.add %3, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : tensor<4x8x80xf64>
// PAD-NEXT:     return %5 : tensor<4x8x80xf64>
// PAD-NEXT:   }
// PAD-NEXT: }

// SPMD:  func.func @main1(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// SPMD-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// SPMD-NEXT:    %1 = stablehlo.custom_call @_SPMDInternalOp_RotateRight(%0) {backend_config = "dimension=2,amount=78", sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// SPMD-NEXT:    return %1 : tensor<4x8x80xf64>
// SPMD-NEXT:  }
// SPMD:  func.func @main3(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// SPMD-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// SPMD-NEXT:    %1 = stablehlo.custom_call @_SPMDInternalOp_RotateRight(%0) {backend_config = "dimension=1,amount=6", sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// SPMD-NEXT:    return %1 : tensor<4x8x80xf64>
// SPMD-NEXT:  }
