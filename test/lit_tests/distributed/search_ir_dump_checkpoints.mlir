// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200" %s -o /dev/null
// RUN: ls %t.dir/testmod-*/Candidates/4/ir | FileCheck %s --check-prefix=IRFILES
// RUN: cat %t.dir/testmod-*/Candidates/4/info.json | FileCheck %s --check-prefix=JSONCOUNTS
// grep -c exits nonzero when it counts 0, which would fail these RUN lines
// under lit's pipefail before FileCheck checks the "0" text; `|| true` leaves
// the exit status to FileCheck alone.
// RUN: (grep -c 'distributed.Collective ' %t.dir/testmod-*/Candidates/4/ir/02-before-collectives.mlir || true) | FileCheck %s --check-prefix=BEFOREIRCOUNT
// RUN: (grep -c 'stablehlo.all_reduce' %t.dir/testmod-*/Candidates/4/ir/03-final.mlir || true) | FileCheck %s --check-prefix=AFTERIRCOUNT
// RUN: cat %t.dir/testmod-*/Candidates/4/ir/01-before-kernel-costing.mlir | FileCheck %s --check-prefix=BEFOREORDER
// RUN: cat %t.dir/testmod-*/Candidates/4/ir/03-final.mlir | FileCheck %s --check-prefix=AFTERORDER

// This module is the same tiny 2-logical-axis fixture
// search_candidate_summary_schedule.mlir uses (reused for its shape only, per
// this suite's convention): search over it is deterministic, and uid 4
// (the eventual winner's parent) is the candidate with a real cross-device
// collective, so it's the interesting one for exercising both the
// collectives-count diff and the post-reorder op-order check below (the
// winner itself, uid 13, fully serializes both logical axes within one
// device and so has no collective to diff or overlap in the first place).

// 00-after-decisions is written by cloneAndApplyDecisions before the
// candidate's lowering pipeline starts; the other three are written from
// inside buildDistributedSearchLoweringPipeline. All four files existing
// shows the dump pass fired at every call site for this candidate.
// IRFILES: 00-after-decisions.mlir
// IRFILES-NEXT: 01-before-kernel-costing.mlir
// IRFILES-NEXT: 02-before-collectives.mlir
// IRFILES-NEXT: 03-final.mlir

// uid 4 has one distributed.Collective (an all-reduce over one of the two
// logical axes) in its abstract form at the 02-before-collectives checkpoint.
// distributed-lower-collectives rewrites it to a stablehlo.async_start/
// all_reduce/async_done chain, so 03-final has one lowered collective and no
// abstract ones. The counts in info.json cover both forms and are
// cross-checked against the two checkpoint IR dumps.
// JSONCOUNTS: "num_collectives_after":1
// JSONCOUNTS: "num_collectives_before":1
// BEFOREIRCOUNT: 1
// AFTERIRCOUNT: 1

// distributed-reorder-schedule runs for every logged candidate
// (reorderForLogging), not only the search winner, so uid 4's final checkpoint
// must already be reordered even though this candidate never finalizes. Its
// two independent dot_generals (one feeding the all-reduce, one unrelated)
// appear at the 01-before-kernel-costing checkpoint with the
// all-reduce-feeding one (the (1x1xf32),(1x1xf32) contraction) second. The
// simulated admission order swaps them, so its transfer can overlap the
// unrelated dot, and the final checkpoint must show the opposite order.
// reorder_schedule.mlir uses the same swap as its signal that reordering ran.
// BEFOREORDER: dot_general %{{.*}}, %{{.*}}, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
// BEFOREORDER: dot_general %{{.*}}, %{{.*}}, contracting_dims = [1] x [0] : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// AFTERORDER: dot_general %{{.*}}, %{{.*}}, contracting_dims = [1] x [0] : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// AFTERORDER: dot_general %{{.*}}, %{{.*}}, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>

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
