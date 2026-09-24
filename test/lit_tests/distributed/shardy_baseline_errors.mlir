// RUN: not enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true --split-input-file %s 2>&1 | FileCheck %s

// Anything the plan expresses that the physical lowering cannot honor is
// rejected with a message naming the value and dimension.

// Two Shardy shardings the analysis must equate disagree on the axis.
// CHECK: error: Shardy baseline: result #0 dim 0 is sharded over [axis1<2,1>], but propagation ties it to argument #0 dim 0 which is sharded over [axis0<2,1>]
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}) {
    %0 = stablehlo.reshape %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8x4xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}

// -----

// A shard of a merged dimension that is minor to an unsharded factor.
// CHECK: error: Shardy baseline: argument #0 dim 0 has a sharded factor ([axis0<2,1>]) minor to an unsharded one; sharded factors must form the major prefix of a dimension
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
    %0 = stablehlo.reshape %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

// A dimension that does not divide by its shard count would need padding.
// CHECK: error: Shardy baseline: argument #0 dim 0 has size 6 which is not divisible by its 4 shards
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}) -> (tensor<6x2xf32>) {
    %0 = stablehlo.negate %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b"}, {}]>]>} : tensor<6x2xf32>
    return %0 : tensor<6x2xf32>
  }
}

// -----

// Sharding the sliced dimension of a slice would need a collective permute.
// CHECK: error: Shardy baseline: argument #0 dim 1 is sharded over [axis0<2,1>], but propagation ties it to a permutation or need-replication factor
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) -> (tensor<8x8xf32>) {
    %0 = stablehlo.slice %x [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : (tensor<8x16xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// A sharding constraint still present after explicit reshard insertion.
// CHECK: error: Shardy baseline: a sharding constraint survived explicit reshard insertion
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x8xf32>) {
    %0 = stablehlo.negate %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }
}

// -----

// Unreduced axes are not supported.
// CHECK: error: Shardy baseline: result #0 has unreduced axes, which are not supported
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}, %w: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x8xf32>) {
    %0 = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a"}>]>} : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
