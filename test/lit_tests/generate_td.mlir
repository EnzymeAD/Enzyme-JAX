// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=1000000 radix=16' | FileCheck %s --check-prefixes=TD,FL4
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=broadcast_reduce' | FileCheck %s --check-prefixes=TD,FL4

// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=10000000 radix=16' | FileCheck %s --check-prefixes=TD,FL64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=cse_concatenate' | FileCheck %s --check-prefixes=TD,FL64

// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=11000000 radix=16' | FileCheck %s --check-prefixes=TD,FL4,FL64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=broadcast_reduce;cse_concatenate' | FileCheck %s --check-prefixes=TD,FL4,FL64

// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=1000000 radix=16' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP4
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=broadcast_reduce' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP4

// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=10000000 radix=16' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=cse_concatenate' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP64

// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=11000000 radix=16' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP4,INTERP64
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=broadcast_reduce;cse_concatenate' --transform-interpreter | FileCheck %s --check-prefixes=INTERPCOMMON,INTERP4,INTERP64

// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td='flags=11000000 radix=16' --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefixes=CLEAN

// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=broadcast_reduce<32>;pad_dot_general(1);iota_simplify<16>(2)' | FileCheck %s --check-prefixes=TD,PARAM

// TD: module attributes {transform.with_named_sequence} {
// TD:  transform.named_sequence @__transform_main(%[[ROOT:.+]]: !transform.any_op) {
// CLEAN-NOT: attributes {transform.with_named_sequence}
// CLEAN-NOT: transform.named_sequence
// TD:    %[[FUNC:.+]] = transform.structured.match ops{["func.func"]} in %[[ROOT]] : (!transform.any_op) -> !transform.any_op
// TD:    transform.apply_patterns to %[[FUNC]] {
// FL4:     transform.apply_patterns.enzyme_hlo.broadcast_reduce
// PARAM:   transform.apply_patterns.enzyme_hlo.broadcast_reduce {benefit = 32 : i64}
// FL64:    transform.apply_patterns.enzyme_hlo.cse_concatenate
// PARAM:   transform.apply_patterns.enzyme_hlo.pad_dot_general postPad = 1
// PARAM:   transform.apply_patterns.enzyme_hlo.iota_simplify {benefit = 16 : i64, parameter = 2 : i64}
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

// INTERPCOMMON-LABEL: @cseconcat
// INTERP64: concatenate
// INTERP64-NOT: concatenate
func.func @cseconcat(%a : tensor<2x3x1xf32>) -> (tensor<2x3x2xf32>, tensor<2x3x2xf32>) {
  %pad = stablehlo.concatenate %a, %a, dim=2 : (tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<2x3x2xf32>
  %pad2 = stablehlo.concatenate %a, %a, dim=2 : (tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<2x3x2xf32>
  return %pad, %pad2 : tensor<2x3x2xf32>, tensor<2x3x2xf32>
}

