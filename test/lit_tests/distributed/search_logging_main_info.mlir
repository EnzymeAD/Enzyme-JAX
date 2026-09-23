// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod commit=deadbeef" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200 dump-candidates" %s -o /dev/null 2> %t.stderr
// RUN: cat %t.dir/testmod-*/Main/info.json | FileCheck %s --check-prefix=INFOJSON
// RUN: FileCheck %s --check-prefix=CANDIDATEUID < %t.stderr
// RUN: grep -c "distributed.logging_candidate_uid" %t.stderr | FileCheck %s --check-prefix=UIDCOUNT

// This module is the same tiny 2-axis fixture search_candidate_uid_lineage.mlir
// uses (reused for its shape only, per this suite's convention). The test
// covers set-distributed-logging feeding distributed-search-strategies, not
// the search decisions themselves. With --set-distributed-logging running
// first, each of the 12 candidates scored for this fixture is stamped with its
// own distributed.logging_candidate_uid before being dumped (dump-candidates),
// and the search's final counts (which candidate won, how many were
// scored/pipeline-errored/infeasible) are written once to Main/info.json.

// llvm::json sorts Main/info.json's keys alphabetically, so this pins the
// exact, complete byte content for this fixture, not a partial match.
// INFOJSON: {"best_candidate_uid":13,"commit_or_version":"deadbeef","input_module_name":"testmod","num_candidates_infeasible":0,"num_candidates_pipeline_errored":0,"num_candidates_scored":12,"num_logical_axes":2}

// Each scored candidate's dumped clone carries its own uid, which varies per
// candidate, so this pins only the attribute's shape. The checkpoint-dump
// passes in the candidate's lowering pipeline have already stamped
// logging_num_collectives_before/_after by the time dumpSearchModule prints
// the clone; their per-candidate values are left unpinned as well.
// CANDIDATEUID: module attributes {distributed.logging_candidate_uid = {{[0-9]+}} : i64, distributed.logging_commit = "deadbeef", distributed.logging_module_name = "testmod", distributed.logging_num_collectives_after = {{[0-9]+}} : i64, distributed.logging_num_collectives_before = {{[0-9]+}} : i64, distributed.logging_root = "{{.*}}/testmod-{{.*}}"}

// One dump per scored candidate.
// UIDCOUNT: 12

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
