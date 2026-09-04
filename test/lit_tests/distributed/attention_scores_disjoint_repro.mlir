// Regression test: reshaping two independently-sharded projections (Q, K)
// splits their shared sequence dimension into the same logical-axis symbol,
// which then merges with each projection's own (distinct) hidden-dim split
// factor into one physical tensor dimension. The dot_general combining Q and
// K^T produces a value whose two non-batch dimensions both contain that
// shared sequence-split symbol, which violates the CastGlobalToLocal
// "disjoint factor groups" invariant. This should compile cleanly once that
// invariant violation is fixed.
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --convert-main-to-distributed-function --materialize-distributed-collectives %s | FileCheck %s

// CHECK: distributed.DistributedYield

module @attention_scores_disjoint_repro {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]>

  func.func @main(
      %arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>},
      %arg1: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg2: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>})
      -> (tensor<8x8x16x16xf32>) {
    // Q and K projections share the same sequence-dim input but use distinct
    // hidden weight matrices, so their hidden-dim reshape splits get distinct
    // axes.
    %q = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %k = stablehlo.dot_general %arg0, %arg2, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>

    // Multi-head reshape to [batch=8, heads=8, seq=16, head_dim=64]. The
    // "seq=16" dimension merges a sequence-split factor (shared by Q and K)
    // with each side's own hidden-split factor.
    %q_heads = stablehlo.reshape %q : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %k_heads = stablehlo.reshape %k : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>

    %k_t = stablehlo.transpose %k_heads, dims = [0, 1, 3, 2] :
      (tensor<8x8x16x64xf32>) -> tensor<8x8x64x16xf32>
    %scores = stablehlo.dot_general %q_heads, %k_t, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] :
      (tensor<8x8x16x64xf32>, tensor<8x8x64x16xf32>) -> tensor<8x8x16x16xf32>

    return %scores : tensor<8x8x16x16xf32>
  }
}
