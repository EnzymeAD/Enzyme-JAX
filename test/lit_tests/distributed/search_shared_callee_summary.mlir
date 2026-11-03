// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate" %s -o /dev/null 2>/dev/null
// RUN: cat %t.dir/testmod-*/Main/info.json | FileCheck %s --check-prefix=MAIN
// RUN: cat %t.dir/testmod-*/Candidates/118/info.json | FileCheck %s --check-prefix=WINNER
// RUN: cat %t.dir/testmod-*/Candidates/118/schedule.json | FileCheck %s --check-prefix=SCHEDULE

// The shared-callee fixture of search_mode_shared_callee_split.mlir: one
// layer function called twice from main. The deterministic beam search's
// winner (uid 118) keeps every compute op inside the callee, so its top-level
// schedule is just the two calls plus zero-cost casts. Its summary still
// reports that compute: the breakdown walks into each call's callee timeline,
// once per call site, so compute_time equals the whole makespan and
// nothing is idle. schedule.json lists each call's callee events in place of
// the call, shifted to the call's start: the second call's matmul starts where
// the first call ends, and depends on the first call's last event.
// MAIN: "best_candidate_uid":118
// SCHEDULE-NOT: "kind":"call"
// SCHEDULE: {"call_stack":["layer"],"description":"compute stablehlo.dot_general: {{.*}}","finish":1024000,"index":4,"kind":"compute","predecessors":[2],"start":0}
// SCHEDULE: {"call_stack":["layer"],"description":"compute distributed.CastGlobalToLocal: {{.*}}","finish":1088000,"index":10,"kind":"compute","predecessors":[9],"start":1088000}
// SCHEDULE: {"call_stack":["layer"],"description":"compute stablehlo.dot_general: {{.*}}","finish":2112000,"index":12,"kind":"compute","predecessors":[10],"start":1088000}
// SCHEDULE: {"call_stack":[],"description":"compute distributed.CastLocalToGlobal: flops=0 bytes=0 time=0","finish":2176000,"index":18,"kind":"compute","predecessors":[17],"start":2176000}
// WINNER: "summary":{"communication_time":0,"compute_time":2176000,"idle_time":0,"total_time":2176000}

module {
  sdy.mesh @mesh = <["tile"=2, "model"=2]>
  func.func private @layer(%h: tensor<8x16xf32>, %w: tensor<16x16xf32>, %cos: tensor<8x4x4xf32>) -> tensor<8x16xf32> {
    %0 = stablehlo.dot_general %h, %w, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
    %1 = stablehlo.reshape %0 : (tensor<8x16xf32>) -> tensor<8x4x4xf32>
    %2 = stablehlo.multiply %1, %cos : tensor<8x4x4xf32>
    %3 = stablehlo.reshape %2 : (tensor<8x4x4xf32>) -> tensor<8x16xf32>
    %4 = stablehlo.add %h, %3 : tensor<8x16xf32>
    return %4 : tensor<8x16xf32>
  }
  func.func @main(%x: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {}]>}, %w0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %w1: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}, %cos: tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) -> tensor<8x16xf32> {
    %0 = func.call @layer(%x, %w0, %cos) : (tensor<8x16xf32>, tensor<16x16xf32>, tensor<8x4x4xf32>) -> tensor<8x16xf32>
    %1 = func.call @layer(%0, %w1, %cos) : (tensor<8x16xf32>, tensor<16x16xf32>, tensor<8x4x4xf32>) -> tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
}
