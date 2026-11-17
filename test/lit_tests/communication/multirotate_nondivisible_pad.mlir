// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{multirotate_custom_call=1 multirotate_spmd=0 dus_sink_slices_comm=0})" %s | FileCheck %s --check-prefix=CCALL
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{multirotate_custom_call=0 multirotate_spmd=1 dus_sink_slices_comm=0})" %s | FileCheck %s --check-prefix=SPMD
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{multirotate_custom_call=1 multirotate_spmd=0 dus_sink_slices_comm=1})" %s | FileCheck %s --check-prefix=SINK

sdy.mesh @mesh = <["x"=2, "y"=2]>

// Non-divisible dim 1 (1519 % 2 == 1) while rotating along dim 2 (3056 % 2 == 0).
func.func @multirotate_nondivisible(%arg0: tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) -> (tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, tensor<4x1519x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) {
  %0:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1519x3056xf32>) -> (tensor<4x1519x3056xf32>, tensor<4x1519x3056xf32>, tensor<4x1519x3056xf32>)
  return %0#0, %0#1, %0#2 : tensor<4x1519x3056xf32>, tensor<4x1519x3056xf32>, tensor<4x1519x3056xf32>
}

// Slice of divisible 4x1520x3056 tensor fed into multi_rotate of 4x1521x3056:
// PadNonDivisibleMultiRotate pads 1521 -> 1522 and PadSinkSlicesComm widens the slice to 1522.
func.func @multirotate_of_slice_sinks(%arg0: tensor<20x1536x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) -> (tensor<4x1521x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, tensor<4x1521x3056xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) {
  %0 = stablehlo.slice %arg0 [8:12, 7:1528, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<20x1536x3056xf32>) -> tensor<4x1521x3056xf32>
  %1:3 = "enzymexla.multi_rotate"(%0) <{dimension = 2 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1521x3056xf32>) -> (tensor<4x1521x3056xf32>, tensor<4x1521x3056xf32>, tensor<4x1521x3056xf32>)
  return %1#0, %1#2 : tensor<4x1521x3056xf32>, tensor<4x1521x3056xf32>
}

// CCALL-LABEL: func.func @multirotate_nondivisible
// CCALL: %[[PAD:.*]] = stablehlo.pad %arg0, %{{.*}}, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] {{.*}} : (tensor<4x1519x3056xf32>, tensor<f32>) -> tensor<4x1520x3056xf32>
// CCALL: %[[ROT:.*]]:3 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[PAD]]) {{.*}} : (tensor<4x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>)
// CCALL: %[[S0:.*]] = stablehlo.slice %[[ROT]]#0 [0:4, 0:1519, 0:3056] {{.*}} : (tensor<4x1520x3056xf32>) -> tensor<4x1519x3056xf32>
// CCALL: %[[S1:.*]] = stablehlo.slice %[[ROT]]#1 [0:4, 0:1519, 0:3056] {{.*}} : (tensor<4x1520x3056xf32>) -> tensor<4x1519x3056xf32>
// CCALL: %[[S2:.*]] = stablehlo.slice %[[ROT]]#2 [0:4, 0:1519, 0:3056] {{.*}} : (tensor<4x1520x3056xf32>) -> tensor<4x1519x3056xf32>
// CCALL: return %[[S0]], %[[S1]], %[[S2]]

// SPMD-LABEL: func.func @multirotate_nondivisible
// SPMD: %[[MPAD:.*]] = stablehlo.pad %arg0, %{{.*}}, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] {{.*}} : (tensor<4x1519x3056xf32>, tensor<f32>) -> tensor<4x1520x3056xf32>
// SPMD: %[[MAN:.*]]:3 = sdy.manual_computation(%[[MPAD]]) {{.*}} (%arg1: tensor<4x760x1528xf32>) {
// SPMD: stablehlo.collective_permute
// SPMD: sdy.return
// SPMD: } : (tensor<4x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>)
// SPMD: stablehlo.slice %[[MAN]]#0 [0:4, 0:1519, 0:3056]

// SINK-LABEL: func.func @multirotate_of_slice_sinks
// SINK-NOT: tensor<4x1521x3056xf32>) -> (tensor<4x1521x3056xf32>
// SINK: stablehlo.custom_call @_SPMDInternalOp_MultiRotate({{.*}}) {{.*}} : (tensor<4x1522x3056xf32>) -> (tensor<4x1522x3056xf32>, tensor<4x1522x3056xf32>, tensor<4x1522x3056xf32>)
