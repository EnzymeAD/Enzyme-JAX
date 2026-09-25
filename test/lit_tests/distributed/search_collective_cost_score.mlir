// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="max-decision-extent=2 beam-size=100 dump-candidates dump-best" %s -o /dev/null 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="max-decision-extent=2 beam-size=100 dump-candidates" %s -o /dev/null 2> %t.first
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --distributed-search-strategies="max-decision-extent=2 beam-size=100 dump-candidates" %s -o /dev/null 2> %t.second
// RUN: diff %t.first %t.second

// The search scores a candidate by the negated makespan a timeline simulation
// predicts for it (ScheduleGraph.h/ScheduleSimulator.h), so the scores are
// deterministic and can be checked by hand. This matmul contracts a dimension
// that can be sharded up to 4 ways; a candidate that keeps the contraction
// distributed needs an all-reduce of the (partial) 64x64 result, and one that
// shards only non-contracting dimensions needs none.
//
// The mesh sets only device_flops = 1e15 and device_mem_bandwidth = 1e18 (SI,
// so F = 1e6 FLOP/ns, M = 1e9 bytes/ns internally: MeshCostMetadata.cpp
// divides every rate attribute by 1e9), leaving axis_bandwidths/
// axis_latencies/launch_latency at MeshCostParams's defaults: bandwidth = 50
// bytes/ns, round latency = 1000ns, launch latency = 5000ns.
//
// A dot_general's roofline time is max(flops/F, bytes/M) (ComputeCost.cpp);
// at this shape bytes/M is negligible next to flops/F, so compute cost is
// effectively 2*M*N*K/F, rounded to the nearest ns (std::llround) when the
// node's isolated duration is first computed:
//
//   - unsharded 64x128 . 128x64:            2*64*64*128/1e6 = 1.048576 -> 1
//   - contracting unsharded, one output dim  2*64*32*128/1e6 = 0.524288 -> 1
//     halved (e.g. 64x32 or 32x64)
//   - fully 4-way sharded, any dim split      2*64*16*128/1e6 = 0.262144 -> 0
//     (e.g. 64x16, 16x64, or 32x32)
//
// The fully-unsharded candidate is a single node with isolatedDuration=1 but
// simulates as makespan=2: the ceilTimeToFinish rounding quirk documented in
// schedule_cost.mlir, where an isolated event's own already-rounded duration
// throttles its simulated rate below 1 and forces its dt up by one. Sharded,
// collective-free candidates instead wrap the local dot_general in
// CastGlobalToLocal/CastLocalToGlobal (zero-cost Movement ops, ComputeCost.h)
// and simulate at exactly their isolatedDuration (3 nodes, no +1).
//
// A 2-way all-reduce of V bytes costs launchLatency + roundLatency*rounds +
// round(V/bandwidth); for a 2-way group recursiveDoubling (1 round) beats
// halvingDoubling (2 rounds), so duration = 5000 + 1000 + round(V/50) =
// 6000 + round(V/50):
//
//   - contracting halved to 64, one output dim also halved (2048-elem,
//     8192-byte partial tile): 6000 + round(8192/50) = 6000 + 164 = 6164;
//     compute (0.262144ns) rounds to 0, so this is the whole makespan.
//   - contracting halved to 64, no other dim sharded (4096-elem, 16384-byte
//     partial tile): 6000 + round(16384/50) = 6000 + 328 = 6328, plus the
//     0.524288ns compute rounded to 1, both on one dependency chain: 6329.
//   - contracting sharded 4 ways: the physical axis's <2,2> x <2,1> factor
//     decomposition (visible as two sub_axis_info collectives in
//     dump-candidates output) makes this two sequential 2-way all-reduces of
//     the full 16384-byte tile rather than one 4-way primitive, so its cost
//     is twice a single hop: 2 * 6328 = 12656 (compute again rounds to 0).
//
// Several collective-free candidates (splitting the output 4 ways in
// different dimension combinations) all round their compute time down to
// 0ns and tie for the best score.

// Candidates with no collective.
// CHECK-DAG: // Search candidate (ok, score=-2.000000e+00):
// CHECK-DAG: // schedule cost: makespan=2 over 1 nodes (communicating=0 computing=2 utilization=5.242880e-01 idle=0)
// CHECK-DAG: // Search candidate (ok, score=-1.000000e+00):
// CHECK-DAG: // schedule cost: makespan=1 over 3 nodes (communicating=0 computing=1 utilization=5.242880e-01 idle=0)
// CHECK-DAG: // Search candidate (ok, score=0.000000e+00):
// CHECK-DAG: // schedule cost: makespan=0 over 3 nodes (communicating=0 computing=0 utilization=0.000000e+00 idle=0)
// A 2-way all-reduce of the 8192-byte 32x64 partial tile: 6164, on the same
// dependency chain as the kernel it follows, so nothing overlaps it.
// CHECK-DAG: // Search candidate (ok, score=-6.164000e+03):
// CHECK-DAG: // schedule cost: makespan=6164 over 7 nodes (communicating=6164 computing=0 utilization=4.252823e-05 idle=6164)
// A 2-way all-reduce of the 16384-byte 64x64 partial tile: 6328, plus 1 of
// compute.
// CHECK-DAG: // Search candidate (ok, score=-6.329000e+03):
// CHECK-DAG: // schedule cost: makespan=6329 over 6 nodes (communicating=6328 computing=1 utilization=8.283900e-05 idle=6328)
// The 4-way contracting split: two sequential 2-way all-reduces, 2 * 6328.
// CHECK-DAG: // Search candidate (ok, score=-1.265600e+04):
// CHECK-DAG: // schedule cost: makespan=12656 over 7 nodes (communicating=12656 computing=0 utilization=2.071302e-05 idle=12656)
// The best candidates need no collective and shard the compute four ways,
// which rounds their compute time down to 0ns.
// CHECK-DAG: // Best search candidate (ok, score=0.000000e+00):
// CHECK-DAG: // schedule cost: makespan=0 over 3 nodes (communicating=0 computing=0 utilization=0.000000e+00 idle=0)

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
