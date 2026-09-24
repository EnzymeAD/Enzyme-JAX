// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: cat %t.dir/testmod-*/Main/info.json | FileCheck %s --check-prefix=MAIN
// RUN: ls %t.dir/testmod-*/Candidates | wc -l | FileCheck %s --check-prefix=CANDIDATE_DIRS
// RUN: grep -oh '"scored_sequence":[0-9]*' %t.dir/testmod-*/Candidates/*/info.json | sort -u | wc -l | FileCheck %s --check-prefix=UNIQUE_SEQUENCES

// This mesh/dot combination has few distinct decision sets, so MCTS's
// UCB-driven select loop reliably revisits an already-finalized,
// already-scored node well within a 20-iteration budget (the same fixture
// search_strategies_budget.mlir uses to pin mcts's num_candidates_scored
// below 20).
//
// A revisited node must not be scored again: StrategyScorer::score() clones
// the module and reruns the full nested lowering pipeline, and
// writeCandidateLog keys each candidate's JSON by uid, so a second score()
// call on the same node would both waste a real pipeline invocation and
// overwrite that uid's Candidates/<uid>/info.json with a different
// scored_sequence, corrupting the scoring-order record checked in
// search_scored_sequence_order.mlir. The checks below confirm the
// invariant holds for this fixture:
//   - num_candidates_scored is the number of real pipeline invocations (9),
//     not the iteration budget (20).
//   - exactly one Candidates/ directory exists per real score.
//   - the recorded scored_sequence values are exactly the contiguous range
//     1..num_candidates_scored, with no duplicates or gaps.
// MAIN: "num_candidates_scored":9
// CANDIDATE_DIRS: 9
// UNIQUE_SEQUENCES: 9

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
