// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --split-input-file --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s --check-prefix=IOTA
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --split-input-file --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s --check-prefix=MASK

// Two attention layers each build the same causal mask from an iota. The
// pipeline merges the copies before it assigns any axes, so one mask feeds
// both layers instead of one value being bound to two layouts.
// IOTA-LABEL: module @shared_causal_mask
// IOTA: stablehlo.iota
// IOTA-NOT: stablehlo.iota
// MASK-LABEL: module @shared_causal_mask
// MASK: stablehlo.compare GE
// MASK-NOT: stablehlo.compare
module @shared_causal_mask {
  func.func public @main(%x: tensor<64x32xf32>, %w1: tensor<32x32xf32>, %w2: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<64x64xf32>
    %i1 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c1 = stablehlo.reshape %i1 : (tensor<64xi32>) -> tensor<64x1xi32>
    %r1 = stablehlo.reshape %i1 : (tensor<64xi32>) -> tensor<1x64xi32>
    %a1 = stablehlo.broadcast_in_dim %c1, dims = [0, 1] : (tensor<64x1xi32>) -> tensor<64x64xi32>
    %b1 = stablehlo.broadcast_in_dim %r1, dims = [0, 1] : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %m1 = stablehlo.compare GE, %a1, %b1, SIGNED : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
    %q1 = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %s1 = stablehlo.dot_general %q1, %x, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
    %p1 = stablehlo.select %m1, %s1, %z : tensor<64x64xi1>, tensor<64x64xf32>
    %y1 = stablehlo.dot_general %p1, %x, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    %i2 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c2 = stablehlo.reshape %i2 : (tensor<64xi32>) -> tensor<64x1xi32>
    %r2 = stablehlo.reshape %i2 : (tensor<64xi32>) -> tensor<1x64xi32>
    %a2 = stablehlo.broadcast_in_dim %c2, dims = [0, 1] : (tensor<64x1xi32>) -> tensor<64x64xi32>
    %b2 = stablehlo.broadcast_in_dim %r2, dims = [0, 1] : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %m2 = stablehlo.compare GE, %a2, %b2, SIGNED : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
    %q2 = stablehlo.dot_general %y1, %w2, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %s2 = stablehlo.dot_general %q2, %y1, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
    %p2 = stablehlo.select %m2, %s2, %z : tensor<64x64xi1>, tensor<64x64xf32>
    %y2 = stablehlo.dot_general %p2, %y1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    return %y2 : tensor<64x32xf32>
  }
}

// -----

// Two layers build the same mask, but propagation shards their copies
// differently, so they get different layouts and must not merge: each layer
// keeps its own compare.
// IOTA-LABEL: module @masks_sharded_differently
// MASK-LABEL: module @masks_sharded_differently
// MASK: stablehlo.compare GE
// MASK: stablehlo.compare GE
module @masks_sharded_differently {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%x: tensor<64x32xf32>, %w1: tensor<32x32xf32>, %w2: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<64x64xf32>
    %i1 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c1 = stablehlo.reshape %i1 : (tensor<64xi32>) -> tensor<64x1xi32>
    %r1 = stablehlo.reshape %i1 : (tensor<64xi32>) -> tensor<1x64xi32>
    %a1 = stablehlo.broadcast_in_dim %c1, dims = [0, 1] : (tensor<64x1xi32>) -> tensor<64x64xi32>
    %b1 = stablehlo.broadcast_in_dim %r1, dims = [0, 1] : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %m1 = stablehlo.compare GE, %a1, %b1, SIGNED : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
    %q1 = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %s1 = stablehlo.dot_general %q1, %x, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
    %p1 = stablehlo.select %m1, %s1, %z : tensor<64x64xi1>, tensor<64x64xf32>
    %k1 = sdy.sharding_constraint %p1 <@mesh, [{"x"}, {}]> : tensor<64x64xf32>
    %y1 = stablehlo.dot_general %k1, %x, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    %i2 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c2 = stablehlo.reshape %i2 : (tensor<64xi32>) -> tensor<64x1xi32>
    %r2 = stablehlo.reshape %i2 : (tensor<64xi32>) -> tensor<1x64xi32>
    %a2 = stablehlo.broadcast_in_dim %c2, dims = [0, 1] : (tensor<64x1xi32>) -> tensor<64x64xi32>
    %b2 = stablehlo.broadcast_in_dim %r2, dims = [0, 1] : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %m2 = stablehlo.compare GE, %a2, %b2, SIGNED : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
    %q2 = stablehlo.dot_general %y1, %w2, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %s2 = stablehlo.dot_general %q2, %y1, contracting_dims = [1] x [1] : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
    %p2 = stablehlo.select %m2, %s2, %z : tensor<64x64xi1>, tensor<64x64xf32>
    %k2 = sdy.sharding_constraint %p2 <@mesh, [{}, {"x"}]> : tensor<64x64xf32>
    %y2 = stablehlo.dot_general %k2, %y1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    return %y2 : tensor<64x32xf32>
  }
}
