// RUN: enzymexlamlir-opt --shardy-to-distributed-pipeline --split-input-file --verify-diagnostics %s

// A reshard point's rule comes straight from a frontend, so a factor shared by
// both sides must sit on the same dimension of each.

// expected-error @+1 {{failed to build module-scoped main logical axis analysis}}
func.func @main(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{enzymexla_reshard_point: expected each factor on one dimension, either shared by both sides or private to one side}}
  %b = stablehlo.custom_call @enzymexla_reshard_point(%x) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=4}, custom>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %b : tensor<4x4xf32>
}

// -----

// expected-error @+1 {{failed to build module-scoped main logical axis analysis}}
func.func @main(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-warning @+2 {{unknown to SDY sharding rule registry}}
  // expected-error @+1 {{enzymexla_reshard_point: expected an sdy.sharding_rule naming the broken dimensions}}
  %b = stablehlo.custom_call @enzymexla_reshard_point(%x) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %b : tensor<4x4xf32>
}
