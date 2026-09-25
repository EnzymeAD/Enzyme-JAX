// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e10, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="max-decision-extent=2 beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMPUTE
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [1.0e6], device_flops = 1.0e21, device_mem_bandwidth = 1.0e21}"' --distributed-search-strategies="max-decision-extent=2 beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NETWORK

// The mesh metadata sets the balance between compute and communication that
// the search optimizes. The matmul is the one of search_collective_cost_score
// (see that file for the roofline/collective cost formulas and the SI-to-ns
// conversion, MeshCostMetadata.cpp's /1e9). Every candidate here has its
// kernel and its collective (if any) on one straight-line dependency chain,
// so the timeline simulator (ScheduleGraph.h/ScheduleSimulator.h) finds
// nothing to overlap and the makespan is the plain sum of the collective and
// compute durations (up to the ceilTimeToFinish +1 quirk documented in
// schedule_cost.mlir: it fires whenever a compute node's rounded
// isolatedDuration underestimates its true roofline time, e.g. 26214.4ns
// rounds down to 26214 below, forcing the simulator to a rate the device
// cannot sustain and so to a makespan of 26215).
//
// Compute-dominant mesh (COMPUTE prefix): slow devices (device_flops = 1e10,
// SI -> F = 10 FLOP/ns) and the default network. Compute per candidate is
// flops / 10, rounded to the nearest ns and then through the +1 quirk above
// when that rounds down:
//   unsharded            1048576 / 10 = 104857.6 -> 104858 (rounds up, exact)
//   two-way sharded       524288 / 10 =  52428.8 -> 52429 (rounds up, exact)
//   four-way sharded      262144 / 10 =  26214.4 -> 26214, +1 quirk -> 26215
// so sharding the compute pays for a collective. Candidates (collective costs
// as derived in search_collective_cost_score.mlir: 6164, 6328, or the
// two-hop 12656):
//   unsharded, no collective                       104858
//   two-way sharded, no collective                   52429
//   two-way sharded + 6328 all-reduce        52429 + 6328 = 58757
//   four-way sharded + 12656 all-reduce      26215 + 12656 = 38871
//   four-way sharded + 6164 all-reduce        26215 + 6164 = 32379
//   four-way sharded, no collective                   26215
// The unsharded candidate is the worst of the collective-free ones and loses
// to every sharded candidate, including ones that communicate. The best is
// still the collective-free four-way sharding, tied across every dimension
// combination that splits the output four ways.
//
// COMPUTE-DAG: // Search candidate (ok, score=-1.048580e+05):
// COMPUTE-DAG: // schedule cost: makespan=104858 over 1 nodes (communicating=0 computing=104858 utilization=9.999962e-01 idle=0)
// COMPUTE-DAG: // Search candidate (ok, score=-5.242900e+04):
// COMPUTE-DAG: // schedule cost: makespan=52429 over 3 nodes (communicating=0 computing=52429 utilization=9.999962e-01 idle=0)
// COMPUTE-DAG: // Search candidate (ok, score=-5.875700e+04):
// COMPUTE-DAG: // schedule cost: makespan=58757 over 6 nodes (communicating=6328 computing=52429 utilization=8.922988e-01 idle=6328)
// COMPUTE-DAG: // Search candidate (ok, score=-3.887100e+04):
// COMPUTE-DAG: // schedule cost: makespan=38871 over 7 nodes (communicating=12656 computing=26215 utilization=6.743948e-01 idle=12656)
// COMPUTE-DAG: // Search candidate (ok, score=-3.237900e+04):
// COMPUTE-DAG: // schedule cost: makespan=32379 over 7 nodes (communicating=6164 computing=26215 utilization=8.096112e-01 idle=6164)
// COMPUTE-DAG: // Search candidate (ok, score=-2.621500e+04):
// COMPUTE-DAG: // schedule cost: makespan=26215 over 3 nodes (communicating=0 computing=26215 utilization=9.999771e-01 idle=0)
// COMPUTE-DAG: // Best search candidate (ok, score=-2.621500e+04):
// COMPUTE-DAG: // schedule cost: makespan=26215 over 4 nodes (communicating=0 computing=26215 utilization=9.999771e-01 idle=0)
//
// Network-dominant mesh (NETWORK prefix): axis_bandwidths = 1e6 (SI) is a
// bandwidth of 0.001 bytes/ns internally, so a step moving V bytes in one
// round costs 6000 + V/0.001 (launchLatency + roundLatency + V/bandwidth),
// and compute at F = 1e12 FLOP/ns rounds every candidate's compute time down
// to 0. At this tiny bandwidth the decomposer's cost-minimizing choice for
// the 4-way collective flips from the two-hop all-reduce pair (the COMPUTE
// case above, or search_collective_cost_score.mlir's mesh) to a
// reduce-scatter/all-reduce/all-gather chain of three quarter-payload
// (8192-byte) steps, since that minimizes total bytes moved rather than
// round count:
//   2-way all-reduce, 8192-byte tile:  6000 + 8192/0.001   =  8198000
//   2-way all-reduce, 16384-byte tile: 6000 + 16384/0.001  = 16390000
//   4-way, three 8192-byte steps:      3 * (6000 + 8192/0.001) = 24594000
// Every candidate with a collective is dominated by it, and the winner needs
// none (score 0, tied across every collective-free sharding).
//
// NETWORK-DAG: // Search candidate (ok, score=-8.198000e+06):
// NETWORK-DAG: // schedule cost: makespan=8198000 over 7 nodes (communicating=8198000 computing=0 utilization=3.197658e-14 idle=8198000)
// NETWORK-DAG: // Search candidate (ok, score=-1.639000e+07):
// NETWORK-DAG: // schedule cost: makespan=16390000 over 6 nodes (communicating=16390000 computing=0 utilization=3.198829e-14 idle=16390000)
// NETWORK-DAG: // Search candidate (ok, score=-2.459400e+07):
// NETWORK-DAG: // schedule cost: makespan=24594000 over 8 nodes (communicating=24594000 computing=0 utilization=1.065886e-14 idle=24594000)
// NETWORK-DAG: // Best search candidate (ok, score=0.000000e+00):
// NETWORK-DAG: // schedule cost: makespan=0 over 1 nodes (communicating=0 computing=0 utilization=0.000000e+00 idle=0)

module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @main(
      %arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<128x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
