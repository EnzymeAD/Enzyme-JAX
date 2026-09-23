// RUN: enzymexlamlir-opt --split-input-file --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=200 dump-candidates" %s -o /dev/null 2>&1 | FileCheck %s

// The search scorer costs a candidate by simulating its event DAG
// (ScheduleGraph.h/ScheduleSimulator.h) to a predicted makespan, so a
// candidate whose work can genuinely run concurrently scores better (a less
// negative makespan) than an equal-total-work candidate whose steps are
// forced to run one after another by a dependency edge.
//
// Both modules below run two 1x2 . 2x1 dot products, each contracting its
// size-2 dimension over a different physical mesh axis ("x" and "y" of a 2x2
// mesh); the non-contracting dimensions are all size 1, so there is no room
// for the search to shard anything else, which keeps the candidate set small
// enough to check by hand. When both contractions are sharded, each needs a
// 2-way all-reduce of a single f32 (4 bytes): isolated duration 4 + 0.11 =
// 4.11 (recursive doubling, one round at the default bandwidth 1, round
// latency 0.01, launch 0.1). Each all-reduce contributes two more nodes than
// its own chain length: distributed-lower-collectives (unconditional for
// every scored candidate) wraps every chain in a zero-cost flatten/
// reshape-back pair, which never affects the makespans computed below (see
// search_collective_cost_score.mlir).
//
// INDEPENDENT (first module): the two dot products have no data dependency,
// so once each one's own all-reduce is ready it can run concurrently with
// the other (they land on different mesh axes here). The makespan is barely
// above one all-reduce's own isolated duration, not their sum: only the
// later-finishing of the two ever extends the critical path.
// CHECK-DAG: // Search candidate (ok, score=-4.110004e+00):
// CHECK-DAG: // schedule cost: makespan=4.110004e+00 over 8 nodes (communicating=4.110002e+00 computing=4.000000e-06 utilization=9.732351e-07 idle=4.110000e+00)

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

// -----

// SERIALIZED (second module): the same two dot products and the same total
// communication (one 2-way all-reduce of a single f32 per contraction), but
// the second dot product now takes the first one's (all-reduced) result,
// broadcast to the right shape, as its own right-hand operand, so it can
// only start once the collective for the first one has finished. That one
// dependency edge is the only difference from the module above: same op
// count, same per-collective payload, same axes available. Because the
// second all-reduce cannot start until the first one finishes, the makespan
// is the plain sum of both all-reduces' isolated durations, 4.11 + 4.11 =
// 8.22, essentially double the INDEPENDENT candidate's makespan above for
// the same total work.
// CHECK-DAG: // Search candidate (ok, score=-8.220004e+00):
// CHECK-DAG: // schedule cost: makespan=8.220004e+00 over 9 nodes (communicating=8.220000e+00 computing=4.008000e-06 utilization=4.866178e-07 idle=8.220000e+00)

module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @main(
      %arg0: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg2: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
      -> (tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    %bc0 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x1xf32>
    %1 = stablehlo.dot_general %arg2, %bc0, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    return %0, %1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
