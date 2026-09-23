// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200" %s -o /dev/null
// RUN: find %t.dir/testmod-*/Candidates -mindepth 1 -maxdepth 1 | wc -l | FileCheck %s --check-prefix=CANDIDATECOUNT
// RUN: find %t.dir/testmod-*/Candidates -name info.json | wc -l | FileCheck %s --check-prefix=INFOCOUNT
// RUN: find %t.dir/testmod-*/Candidates -name schedule.json | wc -l | FileCheck %s --check-prefix=SCHEDULECOUNT
// RUN: cat %t.dir/testmod-*/Candidates/13/info.json | FileCheck %s --check-prefix=WINNERINFO
// RUN: cat %t.dir/testmod-*/Candidates/13/schedule.json | FileCheck %s --check-prefix=WINNERSCHEDULE
// RUN: cat %t.dir/testmod-*/Candidates/4/info.json | FileCheck %s --check-prefix=PARENTINFO

// This module is the same tiny 2-logical-axis fixture
// search_candidate_uid_lineage.mlir uses (reused for its shape only, per this
// suite's convention); search over it is deterministic, and its winner is
// always uid 13 (also pinned by search_logging_main_info.mlir's
// best_candidate_uid check), so hardcoding that uid's directory below is
// stable across runs.

// Every one of the search's 12 scored candidates gets its own
// Candidates/<uid>/ directory with both files, not only the winner or the
// candidates that finalize: a candidate that never becomes anyone's parent
// still helps reconstruct what the search tried.
// CANDIDATECOUNT: 12
// INFOCOUNT: 12
// SCHEDULECOUNT: 12

// uid 13's info.json. llvm::json sorts keys alphabetically, so this pins the
// exact, complete byte content for this fixture's winning candidate, not a
// partial match. heuristic_score is the negation of summary.total_time
// (scoreLowered computes score = 0.0 - result.makespan), so the exact-text pin
// also cross-checks the two literals against each other. This candidate fully
// serializes both logical axes within one device (see the PARENTINFO comment
// below), so no communication remains by the 03-final checkpoint and both
// collectives counts are 0.
// WINNERINFO: {"heuristic_score":-7.9999999999999996e-06,"num_collectives_after":0,"num_collectives_before":0,"parent_uid":4,"pipeline_ok":true,"summary":{"accelerator_compute_time":0,"communication_time":0,"cpu_compute_time":7.9999999999999996e-06,"idle_time":0,"num_act_kernel_matches":0,"total_time":7.9999999999999996e-06},"uid":13}

// uid 13's schedule.json: two independent stablehlo.dot_general compute
// nodes (this candidate fully serializes both logical axes within one
// device, so nothing overlaps and nothing communicates), each 4e-06 long,
// running back to back to the same 7.9999999999999996e-06 makespan
// info.json's summary.total_time reports.
// WINNERSCHEDULE: {"makespan":7.9999999999999996e-06,"nodes":[{"description":"compute stablehlo.dot_general: flops=4 bytes=20 time=4e-06","finish":3.9999999999999998e-06,"index":0,"kind":"compute","predecessors":[],"start":0},{"description":"compute stablehlo.dot_general: flops=4 bytes=20 time=4e-06","finish":7.9999999999999996e-06,"index":1,"kind":"compute","predecessors":[],"start":3.9999999999999998e-06}]}

// uid 13's parent (uid 4) has real communication (one logical axis is
// serialized within a device, the other is sharded across the physical mesh)
// and a much larger idle_time, so this test also covers a candidate with
// communication. Its one collective is still an abstract
// distributed.Collective at the 02-before-collectives checkpoint
// (num_collectives_before=1) and fully lowered to StableHLO primitives by
// 03-final (num_collectives_after=0), as for every other candidate's
// collective in this fixture.
// PARENTINFO: {"heuristic_score":-4.1100019999999997,"num_collectives_after":0,"num_collectives_before":1,"parent_uid":1,"pipeline_ok":true,"summary":{"accelerator_compute_time":0,"communication_time":4.1099999999999994,"cpu_compute_time":6.0000000000000002e-06,"idle_time":4.1099959999999998,"num_act_kernel_matches":0,"total_time":4.1100019999999997},"uid":4}

module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @main(
      %arg0: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg2: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
      %arg3: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
      -> (tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    return %0, %1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
