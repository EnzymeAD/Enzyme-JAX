// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate" %s -o /dev/null 2>/dev/null
// RUN: uid=$(sed -n 's/.*"best_candidate_uid":\([0-9]*\).*/\1/p' %t.dir/testmod-*/Main/info.json) && cat %t.dir/testmod-*/Candidates/$uid/info.json > %t.winner.json && cat %t.dir/testmod-*/Candidates/$uid/schedule.json > %t.schedule.json
// RUN: FileCheck %s --check-prefix=WINNER < %t.winner.json
// RUN: FileCheck %s --check-prefix=SCHEDULE < %t.schedule.json
// RUN: test $(sed -n 's/.*"total_time":\([0-9]*\).*/\1/p' %t.winner.json) -le 2176000

// The shared-callee fixture of search_mode_shared_callee_split.mlir: one
// layer function called twice from main. This checks that a search winner's
// summary and schedule.json cover the compute inside each call's callee,
// once per call site, whichever candidate wins:
//   - the winner is at least as good as keeping every compute op inside the
//     callee (2176000ns, the last RUN line);
//   - its summary reports compute time with nothing idle, so the compute
//     inside the calls is counted (WINNER);
//   - schedule.json lists each call's callee events in place of the call:
//     no call event, and the callee's matmul once per call site (SCHEDULE).
// WINNER: "summary":{"communication_time":{{[0-9]+}},"compute_time":[[TOTAL:[1-9][0-9]*]],"idle_time":0,"total_time":[[TOTAL]]}
// SCHEDULE-NOT: "kind":"call"
// SCHEDULE: {"call_stack":["layer"],"description":"compute stablehlo.dot_general: time={{[0-9]+}}"
// SCHEDULE: {"call_stack":["layer"],"description":"compute stablehlo.dot_general: time={{[0-9]+}}"

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
