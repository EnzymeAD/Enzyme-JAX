// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=4' | FileCheck %s --check-prefixes=TD,FL4
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=64' | FileCheck %s --check-prefixes=TD,FL64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=68' | FileCheck %s --check-prefixes=TD,FL4,FL64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=4' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP4
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=64' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=68' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP4,INTERP64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=68' --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefixes=CLEAN

// TD: module attributes {transform.with_named_sequence} {
// TD:  transform.named_sequence @__transform_main(%[[ROOT:.+]]: !transform.any_op) {
// CLEAN-NOT: transform.named_sequence
// TD:    %[[FUNC:.+]] = transform.structured.match ops{["func.func"]} in %[[ROOT]] : (!transform.any_op) -> !transform.any_op
// TD:    transform.apply_patterns to %[[FUNC]] {
// FL4:     transform.apply_patterns.enzyme_hlo.broadcast_reduce
// FL64:    transform.apply_patterns.enzyme_hlo.reduce_pad
// TD:    } : !transform.any_op
// TD:    transform.yield
// TD:  }

// INTERPCOMMON-LABEL: @broadcastreduce
// INTERP4: reduce
// INTERP4: convert
// INTERP4: multiply
func.func @broadcastreduce(%154: tensor<1x3072xf32>, %151: tensor<f32>) -> tensor<f32> {
  %211 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<1x3072x32xf32>
  %212 = stablehlo.reduce(%211 init: %151) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x3072x32xf32>, tensor<f32>) -> tensor<f32>
  return %212 : tensor<f32>
}

// INTERPCOMMON-LABEL: @reducepad
// INTERP64: reduce
// INTERP64: pad
func.func @reducepad(%a : tensor<2x3x1xf32>, %b : tensor<f32>) -> tensor<6x1xf32> {
  %pv = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %pad = stablehlo.pad %a, %pv, low = [1, 2, 0], high = [3, 4, 0], interior = [0, 1, 0] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<6x11x1xf32>
  %conv = stablehlo.reduce(%pad init: %b) applies stablehlo.add across dimensions = [1] : (tensor<6x11x1xf32>, tensor<f32>) -> tensor<6x1xf32>
  return %conv : tensor<6x1xf32>
}

