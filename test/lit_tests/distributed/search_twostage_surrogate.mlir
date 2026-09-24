// RUN: rm -rf %t.random && mkdir -p %t.random
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.random module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=random search-samples=200 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: rm -rf %t.nopool && mkdir -p %t.nopool
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.nopool module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=200 surrogate-train=50 surrogate-pool=150 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: find %t.random/testmod-*/Candidates -name info.json | xargs grep -oh "\"heuristic_score\":[^,]*" | sort > %t.random.scores
// RUN: find %t.nopool/testmod-*/Candidates -name info.json | xargs grep -oh "\"heuristic_score\":[^,]*" | sort > %t.nopool.scores
// RUN: diff %t.random.scores %t.nopool.scores
// RUN: rm -rf %t.bigpool && mkdir -p %t.bigpool
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.bigpool module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=200 surrogate-train=50 surrogate-pool=3000 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: cat %t.bigpool/testmod-*/Main/info.json | FileCheck %s

// With the pool equal to K there is nothing to select from (every rollout is
// kept, see runTwoStage's `keep.size() < K` branch in
// SearchStrategyRunners.h), so two-stage scores exactly the leaves random
// would: the rng draws the same 200-leaf sequence either way (scoring never
// consumes rng state), split into 50 training + 150 pool instead of 200
// straight rollouts. Comparing every scored candidate's heuristic_score (not
// just the winner) makes sure a broken pool selection would show up even if
// both runs happened to agree on the best candidate by coincidence.
//
// With a much larger pool (surrogate-pool=3000), two-stage should still find
// a viable winner: this just checks the run completes and scores its exact
// budget, without pinning which candidate wins. Which candidate wins is not
// this test's concern; search_strategies_budget.mlir already pins the exact
// budget, and this pool size differs from that test only in how many
// candidates get ranked but never scored.
// CHECK: "num_candidates_scored":200

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
