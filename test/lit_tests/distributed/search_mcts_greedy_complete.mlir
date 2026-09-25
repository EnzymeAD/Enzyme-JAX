// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir/def module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir/neg module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1 mcts-greedy-complete-after=-1" %s -o /dev/null 2>/dev/null
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir/g0 module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1 mcts-greedy-complete-after=0" %s -o /dev/null 2>/dev/null
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir/g1 module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1 mcts-greedy-complete-after=1" %s -o /dev/null 2>/dev/null
// RUN: diff %t.dir/def/testmod-*/Main/info.json %t.dir/neg/testmod-*/Main/info.json
// RUN: cat %t.dir/def/testmod-*/Main/info.json | FileCheck %s --check-prefix=DEFAULT
// RUN: cat %t.dir/g0/testmod-*/Main/info.json | FileCheck %s --check-prefix=GREEDY0
// RUN: cat %t.dir/g1/testmod-*/Main/info.json | FileCheck %s --check-prefix=GREEDY1
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=2 mcts-greedy-complete-after=-2" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT

// mcts-greedy-complete-after=-1 (the default) keeps rollouts fully random, so
// an explicit -1 must match the default run byte for byte, and both keep the
// counts pinned by search_mcts_revisit_reuses_score.mlir. With N >= 0 each
// rollout completes greedily after N random steps. N=0 is deterministic given
// the seed; the completed leaf is cached per selected node, so revisits still
// reuse a score. Every scored greedy leaf is a finalized candidate that gets a
// logged directory.

// DEFAULT: "num_candidates_scored":9
// GREEDY0: "best_candidate_uid":
// GREEDY0-SAME: "num_candidates_pipeline_errored":0
// GREEDY1: "best_candidate_uid":
// GREEDY1-SAME: "num_candidates_pipeline_errored":0
// REJECT: mcts-greedy-complete-after must be -1

module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(
      %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> (tensor<8x8xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
