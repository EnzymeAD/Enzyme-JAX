// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="beam-size=200 act-patched-module-file=%S/act_patched_merged_dot_kernel.mlir.txt" %s -o /dev/null 2>/dev/null
// RUN: cat %t.dir/testmod-*/Candidates/13/schedule.json | FileCheck %s

// This module is the same tiny 2-logical-axis fixture
// search_candidate_summary_schedule.mlir uses, with act-patched-module-file
// forcing every scored candidate's (merged, see
// act_patched_merged_dot_kernel.mlir.txt) kernel body to be spliced with an
// ACT-dispatched launch instead of running its two stablehlo.dot_generals
// directly. This is a test-only seam standing in for a real ACT/Docker
// dispatch (see LowerKernelsToExecutablePass's own act-patched-module-file
// option, forwarded here from distributed-search-strategies). Search over
// this fixture is deterministic and its winner is always uid 13, same as
// search_candidate_summary_schedule.mlir.
//
// schedule.json's per-node "engine" field is what a timeline visualization
// uses to color an ACT accelerator dispatch apart from ordinary CPU-costed
// compute: the spliced func.call to @act_launch (the only op carrying
// act.cost) is engine=act and its description carries the launch's
// act.kernel name, while every other node in the same schedule
// (alloc/subview/stage/fetch calls, none of which carry act.cost) is
// engine=cpu.
// CHECK: "description":"compute memref.alloc: flops=0 bytes=0 time=0","engine":"cpu"
// CHECK: act_kernel=merged_dot_1x2_2x1","engine":"act"
// CHECK-NOT: act_kernel

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
