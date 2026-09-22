// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e1, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMPUTE
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [1.0e-3], device_flops = 1.0e12, device_mem_bandwidth = 1.0e12}"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NETWORK

// The mesh metadata sets the balance between compute and communication that
// the search optimizes. The matmul is the one of search_collective_cost_score:
// its local roofline time is 2 * M * N * K / device_flops when memory traffic
// is negligible, and its network costs are those listed there (V + 0.11 per
// step at bandwidth 1). Every candidate here has its kernel and its
// collective (if any) on one straight-line dependency chain, so the timeline
// simulator (ScheduleGraph.h/ScheduleSimulator.h) finds nothing to overlap
// and the makespan is the plain sum of the collective and compute durations.
//
// Compute-dominant mesh (COMPUTE prefix): slow devices (device_flops = 10) and
// the default network. Compute per candidate is flops / 10:
//   unsharded            1048576 / 10 = 104857.6
//   two-way sharded       524288 / 10 =  52428.8
//   four-way sharded      262144 / 10 =  26214.4
// so sharding the compute pays for a collective. Candidates:
//   unsharded, no collective                     104857.6
//   two-way sharded, no collective                52428.8
//   two-way sharded + 16384.11 all-reduce         52428.8 + 16384.11 = 68812.91
//   four-way sharded + 24576.33 all-reduce        26214.4 + 24576.33 = 50790.73
//   four-way sharded + 8192.11 all-reduce         26214.4 +  8192.11 = 34406.51
//   four-way sharded, no collective               26214.4
// The unsharded candidate is the worst of the collective-free ones and loses
// to every sharded candidate, including ones that communicate. The best is
// the collective-free four-way sharding.
//
// COMPUTE-DAG: // Search candidate (ok, score=-1.048576e+05):
// COMPUTE-DAG: // schedule cost: makespan=1.048576e+05 over 1 nodes (communicating=0.000000e+00 computing=1.048576e+05 utilization=1.000000e+00 idle=0.000000e+00)
// COMPUTE-DAG: // Search candidate (ok, score=-5.242880e+04):
// COMPUTE-DAG: // schedule cost: makespan=5.242880e+04 over 1 nodes (communicating=0.000000e+00 computing=5.242880e+04 utilization=1.000000e+00 idle=0.000000e+00)
// COMPUTE-DAG: // Search candidate (ok, score=-6.881291e+04):
// COMPUTE-DAG: // schedule cost: makespan=6.881291e+04 over 2 nodes (communicating=1.638411e+04 computing=5.242880e+04 utilization=7.619035e-01 idle=1.638411e+04)
// COMPUTE-DAG: // Search candidate (ok, score=-5.079073e+04):
// COMPUTE-DAG: // schedule cost: makespan=5.079073e+04 over 4 nodes (communicating=2.457633e+04 computing=2.621440e+04 utilization=5.161257e-01 idle=2.457633e+04)
// COMPUTE-DAG: // Search candidate (ok, score=-3.440651e+04):
// COMPUTE-DAG: // schedule cost: makespan=3.440651e+04 over 2 nodes (communicating=8.192110e+03 computing=2.621440e+04 utilization=7.619023e-01 idle=8.192110e+03)
// COMPUTE-DAG: // Search candidate (ok, score=-2.621440e+04):
// COMPUTE-DAG: // schedule cost: makespan=2.621440e+04 over 1 nodes (communicating=0.000000e+00 computing=2.621440e+04 utilization=1.000000e+00 idle=0.000000e+00)
// COMPUTE-DAG: // Best search candidate (ok, score=-2.621440e+04):
//
// Network-dominant mesh (NETWORK prefix): an axis bandwidth of 0.001 makes a
// collective cost V / 0.001 + 0.11 instead of V + 0.11, so the 8192-byte all-reduce costs 8192000.11 and the
// 4-way one 24576000.33, and compute at 1e12 FLOP per time unit is negligible
// (2.62144e-07 for the four-way sharded matmul). Every candidate with a
// collective is dominated by it, and the winner needs none.
//
// NETWORK-DAG: // Search candidate (ok, score=-8.192000e+06):
// NETWORK-DAG: // schedule cost: makespan=8.192000e+06 over 2 nodes (communicating=8.192000e+06 computing=2.621440e-07 utilization=3.200000e-14 idle=8.192000e+06)
// NETWORK-DAG: // Search candidate (ok, score=-1.638400e+07):
// NETWORK-DAG: // schedule cost: makespan=1.638400e+07 over 2 nodes (communicating=1.638400e+07 computing=5.242880e-07 utilization=3.200000e-14 idle=1.638400e+07)
// NETWORK-DAG: // Search candidate (ok, score=-2.457600e+07):
// NETWORK-DAG: // schedule cost: makespan=2.457600e+07 over 4 nodes (communicating=2.457600e+07 computing=2.621440e-07 utilization=1.066667e-14 idle=2.457600e+07)
// NETWORK-DAG: // Best search candidate (ok, score=-2.621440e-07):
// NETWORK-DAG: // schedule cost: makespan=2.621440e-07 over 1 nodes (communicating=0.000000e+00 computing=2.621440e-07 utilization=1.000000e+00 idle=0.000000e+00)

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
