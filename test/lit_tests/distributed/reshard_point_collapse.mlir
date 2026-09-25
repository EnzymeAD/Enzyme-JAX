// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" --mlir-print-ir-after=merge-adjacent-trivial-kernels -o /dev/null 2>&1 | FileCheck %s --check-prefix=MERGED
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate dump-best" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=SEARCH

// A reshard point whose two sides end up sharded alike. Its only dimension is
// broken, so the distributed IR still carries a collective between the two
// sides' axes, but once both are bound to the same physical factors that
// collective is an identity: lowering drops it and the kernels on either side
// merge back into one.

module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%x: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data", "model"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data", "model"}]>}) {
    %e = stablehlo.exponential %x : tensor<16xf32>
    %b = stablehlo.custom_call @enzymexla_reshard_point(%e) {sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([j]) {i=16, j=16}, custom>} : (tensor<16xf32>) -> tensor<16xf32>
    %n = stablehlo.negate %b : tensor<16xf32>
    return %n : tensor<16xf32>
  }
}

// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.exponential
// CHECK: distributed.Collective %{{.*}} : tensor<1xf32> on %{{.*}} : <16> to tensor<1xf32> on %{{.*}} : <16> reduces ()
// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.negate
// CHECK-NOT: enzymexla_reshard_point

// MERGED: IR Dump After MergeAdjacentTrivialKernelsPass
// MERGED-NOT: distributed.Collective
// MERGED: distributed.DistributedKernel (%{{.*}} : tensor<4xf32>)
// MERGED: stablehlo.exponential %{{.*}} : tensor<4xf32>
// MERGED-NEXT: stablehlo.negate %{{.*}} : tensor<4xf32>
// MERGED-NOT: distributed.DistributedKernel
// MERGED-NOT: distributed.Collective
// MERGED: distributed.PhysicalMesh

// Search keeps the whole chain split four ways with no collective.
// SEARCH: Best search candidate (ok
// SEARCH: distributed.CastGlobalToLocal %arg0 {{.*}} : tensor<16xf32> -> tensor<4xf32>
// SEARCH-NEXT: stablehlo.exponential %{{.*}} : tensor<4xf32>
// SEARCH-NEXT: stablehlo.negate %{{.*}} : tensor<4xf32>
// SEARCH-NEXT: distributed.CastLocalToGlobal
