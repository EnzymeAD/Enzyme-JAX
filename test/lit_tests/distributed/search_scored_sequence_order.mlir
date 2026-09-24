// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=random search-samples=8 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: find %t.dir/testmod-*/Candidates -name info.json | xargs grep -oh "\"scored_sequence\":[0-9]*" | cut -d: -f2 > %t.seqs
// RUN: find %t.dir/testmod-*/Candidates -name info.json | xargs grep -oh "\"scored_at_ms\":[0-9]*" | cut -d: -f2 > %t.times
// RUN: paste %t.seqs %t.times | sort -n -k1,1 > %t.pairs
// RUN: cut -f1 %t.pairs | FileCheck %s --check-prefix=SEQ
// RUN: awk 'BEGIN {ok=1} {if ($2 < prev) ok=0; prev=$2} END {print (ok ? "TIME_OK" : "TIME_BAD")}' %t.pairs | FileCheck %s --check-prefix=TIME

// Every candidate's info.json records scored_sequence/scored_at_ms (see
// StrategyScorer::score, the single point every search strategy funnels
// through), so a convergence-vs-scoring-order curve can be reconstructed for
// any search strategy afterward, including this file's random strategy.
//
// %t.pairs holds one "scored_sequence scored_at_ms" line per candidate,
// sorted numerically by scored_sequence. The SEQ checks below confirm the
// sorted scored_sequence column reads 1, 2, ..., 8: a duplicate value would
// repeat a line and a gap would break the run, so this confirms
// scored_sequence is a genuine permutation of 1..8.
// SEQ: 1
// SEQ-NEXT: 2
// SEQ-NEXT: 3
// SEQ-NEXT: 4
// SEQ-NEXT: 5
// SEQ-NEXT: 6
// SEQ-NEXT: 7
// SEQ-NEXT: 8

// scored_at_ms must be non-decreasing once ordered by scored_sequence.
// TIME: TIME_OK

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
