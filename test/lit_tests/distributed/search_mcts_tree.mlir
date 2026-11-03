// RUN: rm -rf %t.a && mkdir -p %t.a
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.a module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=32 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: rm -rf %t.b && mkdir -p %t.b
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.b module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=32 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: diff %t.a/testmod-*/Main/info.json %t.b/testmod-*/Main/info.json
// RUN: cat %t.a/testmod-*/Main/info.json | FileCheck %s

// MCTS keeps its per-node statistics in a llvm::DenseMap<const Node *,
// UctStats<Node>>, which moves every value when it grows, so nothing may hold
// a reference to one across a lookup. This input is large enough for the map
// to pass that point: each iteration registers its rollout leaf and the
// children selection scans, so a budget of 32 registers about 70 entries,
// past the 47 that DenseMap's first allocation of 64 buckets holds. That
// exercises the growth the select/backprop loop in
// SearchStrategyRunners.h::runMcts is written to survive (see the comment
// above UctStats there).
//
// What this test actually pins is determinism at a fixed seed (both runs'
// Main/info.json, including which candidate won, must be byte-identical) on
// a tree deep enough to exercise the select path across a rehash.

// CHECK: "best_candidate_uid":
// CHECK-SAME: "num_candidates_scored":32

module @attention {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]>
  func.func @main(
      %arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>},
      %arg1: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg2: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>})
      -> (tensor<8x8x16x16xf32>) {
    %q = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %k = stablehlo.dot_general %arg0, %arg2, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %q_heads = stablehlo.reshape %q : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %k_heads = stablehlo.reshape %k : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %k_t = stablehlo.transpose %k_heads, dims = [0, 1, 3, 2] :
      (tensor<8x8x16x64xf32>) -> tensor<8x8x64x16xf32>
    %scores = stablehlo.dot_general %q_heads, %k_t, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] :
      (tensor<8x8x16x64xf32>, tensor<8x8x64x16xf32>) -> tensor<8x8x16x16xf32>
    return %scores : tensor<8x8x16x16xf32>
  }
}
