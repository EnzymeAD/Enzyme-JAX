// RUN: rm -rf %t.dir %t.baseline.mlir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s -o %t.baseline.mlir 2> /dev/null
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="beam-size=200 baseline-module=%t.baseline.mlir" %s -o /dev/null 2> /dev/null
// RUN: cat %t.dir/testmod-*/Baseline/info.json | FileCheck %s --check-prefix=BINFO
// RUN: cat %t.dir/testmod-*/Baseline/schedule.json | FileCheck %s --check-prefix=BSCHED
// RUN: ls %t.dir/testmod-*/Baseline/ir | FileCheck %s --check-prefix=BIR
// RUN: cat %t.dir/testmod-*/Main/info.json | FileCheck %s --check-prefix=MAIN
// RUN: ls %t.dir/testmod-*/Candidates | FileCheck %s --check-prefix=CANDS --implicit-check-not={{[A-Za-z]}}
// RUN: not grep -rlE "logging_baseline|shardy_baseline|baseline_score" %t.dir/testmod-*/Candidates
// RUN: rm -rf %t.nb.dir && mkdir -p %t.nb.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.nb.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="beam-size=200" %s -o /dev/null 2> /dev/null
// RUN: test ! -e %t.nb.dir/testmod-*/Baseline
// RUN: cat %t.nb.dir/testmod-*/Main/info.json | FileCheck %s --check-prefix=NOMAIN
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="beam-size=200 baseline-module=%s" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOATTR
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="beam-size=200 baseline-module=%t.missing.mlir" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING

// The Shardy expert plan, lowered by the frontend with preserve-shardy-sharding
// and handed to the search as baseline-module, is scored through the same
// lowering pipeline as a candidate but logged under Baseline/, with the same
// file layout as a candidate directory minus the uid fields.
// BINFO: "heuristic_score":-{{[0-9.e+]+}}
// BINFO-SAME: "pipeline_ok":true
// BINFO-SAME: "summary":{{.*}}"total_time":{{[0-9.e+]+}}
// BINFO-NOT: "uid"
// BSCHED: "makespan":{{[0-9.e+]+}}
// BSCHED-SAME: "nodes":[
// BIR: 00-after-decisions.mlir
// BIR: 03-final.mlir

// Main/info.json points at the baseline; the candidate counts are those of the
// same search without a baseline (see search_logging_main_info.mlir).
// MAIN: "baseline_dir":"Baseline","baseline_score":-{{[0-9.e+]+}}
// MAIN-SAME: "num_candidates_infeasible":0,"num_candidates_pipeline_errored":0,"num_candidates_scored":12

// Candidates/ holds candidate uids only, and nothing in it mentions the
// baseline.
// CANDS: {{[0-9]+}}

// Without a baseline, Main/info.json has no baseline fields.
// NOMAIN-NOT: baseline

// A baseline that was not produced by the preserving frontend, or that cannot
// be read, fails the pass rather than being skipped.
// NOATTR: error: distributed-search-strategies: baseline-module '{{.*}}' was not produced by shardy-to-distributed-pipeline with preserve-shardy-sharding=true
// MISSING: error

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
