// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200" %s -o /dev/null
// RUN: cat %t.dir/testmod-*/Main/logical_axes.json %t.dir/testmod-*/Main/op_axis_tags.json | FileCheck %s

// This module is the same tiny 2-axis fixture search_candidate_uid_lineage.mlir
// uses (reused for its shape only). The property under test is the def-chain
// resolution in op_axis_tags.json: a tagged op's dim_partitioning_axes index
// must resolve to the logging_id of the LogicalMeshAxesOp that produces it.
//
// cluster-distributed-kernels puts each dot_general in its own
// DistributedKernelOp (they are structurally independent), so each kernel's
// partitioning_axes list has exactly one entry and every raw
// dim_partitioning_axes index is 0. A resolver that skipped the def-chain and
// re-emitted the raw index would print 0 for both ops, so the checks compare
// against the real logical_axes.json ids of both axes.
//
// logical_axes.json assigns ids in module walk order. ConvertMainToDistributed
// Function creates "x"'s LogicalMeshAxesOp before "y"'s, since %arg0's
// sharding (dim 1 on "x") is seen before %arg2's (dim 1 on "y"), so x's id is
// listed first (XID below) and y's second (YID). The first dot_general (%0,
// from %arg0/%arg1) is sharded on "x", [{}, {"x"}]/[{"x"}, {}], so its tagged
// dim must resolve to XID. The second (%1, from %arg2/%arg3) is sharded on
// "y", [{}, {"y"}]/[{"y"}, {}], so its tagged dim must resolve to YID.
//
// A literal "[[" or "]]", which nested JSON arrays contain, reads to
// FileCheck as its own [[...]] capture delimiter, so every literal bracket
// below is spelled {{\[}} or {{\]}}. The location field's wildcard is
// {{[^)]*}}: a greedy {{.*}} would match through the last ")" in the whole
// two-op stream and skip the second op's check.
// CHECK: "axes":[{"extent":2,"kind":"logical_mesh","logging_id":[[XID:[0-9]+]]},{"extent":2,"kind":"logical_mesh","logging_id":[[YID:[0-9]+]]}]
// CHECK: "argument_shardings":{{\[}}{"dims":{{\[}}{{\[}}{{\]}},{{\[}}[[XID]]{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}},{"dims":{{\[}}{{\[}}[[XID]]{{\]}},{{\[}}{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}}{{\]}},"location":"loc({{[^)]*}})","output_shardings":{{\[}}{"dims":{{\[}}{{\[}}{{\]}},{{\[}}{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}}{{\]}}}
// CHECK: "argument_shardings":{{\[}}{"dims":{{\[}}{{\[}}{{\]}},{{\[}}[[YID]]{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}},{"dims":{{\[}}{{\[}}[[YID]]{{\]}},{{\[}}{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}}{{\]}},"location":"loc({{[^)]*}})","output_shardings":{{\[}}{"dims":{{\[}}{{\[}}{{\]}},{{\[}}{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}}{{\]}}}

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
