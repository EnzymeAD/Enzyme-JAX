// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="beam-size=200" %s -o /dev/null 2>/dev/null
// RUN: cat %t.dir/testmod-*/Main/op_axis_tags.json | FileCheck %s --check-prefix=TAGS
// RUN: cat %t.dir/testmod-*/Main/info.json | FileCheck %s --check-prefix=MAIN
// RUN: cat %t.dir/testmod-*/Candidates/3/info.json | FileCheck %s --check-prefix=WINNERINFO

// A constant shared by two independent dot_generals is the shape
// SinkKernelBoundaryValues.cpp describes: greedy folding commons an
// identical constant just above the kernels that use it. Here
// cluster-distributed-kernels merges both dot_generals into one
// merge-adjacent DistributedKernel, duplicating the constant into the
// merged body; a non-merge-adjacent pair of kernels would produce the same
// shared-constant shape upstream of that merge instead. Either way, this
// exercises op_axis_tags.json and axis_bindings (SearchStrategies.cpp)
// walking a module containing a stablehlo.constant, an op that never
// participates in a real cross-device reshard, unlike every other tagged op
// here.
//
// This doubles as a "does it crash" test: op_axis_tags.json and
// axis_bindings both walk ops looking for
// distributed.argument_shardings/output_shardings and must degrade
// gracefully (skip, don't assert) on any op that lacks them, since a
// constant may or may not carry them depending on where
// cluster-distributed-kernels leaves it relative to a kernel boundary. A
// crash or nonzero exit from the RUN line would fail the test before any
// FileCheck below runs.

// One shared logical axis ("x"), so exactly one axis_bindings entry per
// candidate. This candidate (uid 3, the winner: both dot_generals fully
// serialized within one device) binds it to a device_local_axis factor, the
// same "no communication" shape search_candidate_summary_schedule.mlir's
// WINNERINFO covers for a two-logical-axis fixture. Here the same lookup
// runs alongside a shared stablehlo.constant rather than in isolation.
// MAIN: "num_candidates_scored":2
// WINNERINFO: "axis_bindings":{{\[}}{"logging_id":1,"physical_factors":{{\[}}"!axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>"{{\]}}}{{\]}}

// op_axis_tags.json records the constant too: it carries
// distributed.output_shardings inside the merged kernel body, since
// cluster-distributed-kernels shards it identically on every device, and an
// empty argument_shardings array since it takes no operands.
// TAGS: {"argument_shardings":{{\[}}{{\]}},"location":"loc({{[^)]*}})","output_shardings":{{\[}}{"dims":{{\[}}{{\[}}1{{\]}},{{\[}}{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}}{{\]}}}

module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(
      %arg0: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> (tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %cst = stablehlo.constant dense<1.0> : tensor<2x1xf32>
    %0 = stablehlo.dot_general %arg0, %cst, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    %1 = stablehlo.dot_general %arg1, %cst, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    return %0, %1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
