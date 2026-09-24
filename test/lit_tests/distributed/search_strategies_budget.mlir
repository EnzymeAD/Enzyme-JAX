// RUN: rm -rf %t.rnd && mkdir -p %t.rnd
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.rnd module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=random search-samples=20 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: cat %t.rnd/testmod-*/Main/info.json | FileCheck %s --check-prefix=BUDGET20
// RUN: rm -rf %t.mcts && mkdir -p %t.mcts
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.mcts module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: cat %t.mcts/testmod-*/Main/info.json | FileCheck %s --check-prefix=MCTS_BUDGET
// RUN: rm -rf %t.ts && mkdir -p %t.ts
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.ts module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=20 surrogate-train=5 search-seed=1" %s -o /dev/null 2>/dev/null
// RUN: cat %t.ts/testmod-*/Main/info.json | FileCheck %s --check-prefix=BUDGET20
// RUN: rm -rf %t.fb && mkdir -p %t.fb
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.fb module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=20 surrogate-train=0 search-seed=1" %s -o /dev/null 2> %t.fb.stderr
// RUN: cat %t.fb/testmod-*/Main/info.json | FileCheck %s --check-prefix=BUDGET20
// RUN: FileCheck %s --check-prefix=FALLBACK < %t.fb.stderr
// RUN: rm -rf %t.pool && mkdir -p %t.pool
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.pool module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=20 surrogate-train=5 surrogate-pool=5 search-seed=1" %s -o /dev/null 2> %t.pool.stderr
// RUN: cat %t.pool/testmod-*/Main/info.json | FileCheck %s --check-prefix=BUDGET20
// RUN: FileCheck %s --check-prefix=POOL < %t.pool.stderr
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=bogus" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts uct-backup=bogus" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BACKUP
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage surrogate-target=bogus" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET

// Every strategy spends exactly its budget, counted by the search's own
// candidate count (Main/info.json's num_candidates_scored, the same count
// StrategyScorer::score() increments on every call regardless of which
// strategy is driving it).
// BUDGET20: "num_candidates_scored":20

// MCTS is the exception: its UCB-driven select loop can revisit a finalized
// node it already scored in an earlier iteration (this tiny tree has few
// distinct decision sets relative to the 20-iteration budget, so it happens
// here). A revisit reuses the score already recorded on that node instead of
// running the lowering pipeline again, so num_candidates_scored (real
// pipeline invocations) comes in under the iteration budget. See
// search_mcts_revisit_reuses_score.mlir for the invariant this protects: a
// revisited decision set never gets a second lowering pass or overwrites its
// candidate log entry.
// MCTS_BUDGET: "num_candidates_scored":9

// With no training samples the surrogate cannot be fitted, and two-stage
// falls back to random rollouts for the whole budget.
// FALLBACK: surrogate fit failed on 0 samples; remaining 20 evaluations fall back to random rollouts

// A pool smaller than the candidates left to evaluate would leave budget
// unspent, so it is raised and the run still spends all 20.
// POOL: surrogate-pool 5 is below the 15 candidates to evaluate; raising it to 15

// UNKNOWN: unknown search-strategy 'bogus' (known: beam, random, mcts, twostage)

// Both are compared against one spelling, so an unknown value has to fail
// rather than silently selecting the default.
// BACKUP: unknown uct-backup 'bogus' (known: mean, max)
// TARGET: unknown surrogate-target 'bogus' (known: log, linear)

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
