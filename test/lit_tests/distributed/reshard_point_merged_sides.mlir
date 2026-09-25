// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline=dump-value-axes=true %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DUMP
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate" %s 2>/dev/null | FileCheck %s --check-prefix=SEARCH

// A reshard point whose two sides are tied together anyway: the residual add
// consumes both the pre-break value %e and the post-break value %n, so the
// token axes on either side of the break unify. No collective remains at the
// break and the whole chain is one kernel.

module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
    %e = stablehlo.exponential %x : tensor<8x4xf32>
    %b = stablehlo.custom_call @enzymexla_reshard_point(%e) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, j]) {i=8, j=4, k=8}, custom>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
    %n = stablehlo.negate %b : tensor<8x4xf32>
    %s = stablehlo.add %e, %n : tensor<8x4xf32>
    return %s : tensor<8x4xf32>
  }
}

// DUMP: %arg0 : {{\[}}[[T:%[0-9]+]]{{\]}} {{\[}}[[D:%[0-9]+]]{{\]}}
// DUMP-COUNT-4: %{{[0-9]+}} : {{\[}}[[T]]{{\]}} {{\[}}[[D]]{{\]}}
// DUMP-NEXT: ===== End Logical Axes per Value =====

// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.exponential
// CHECK-NEXT: stablehlo.negate
// CHECK-NEXT: stablehlo.add
// CHECK-NOT: distributed.Collective
// CHECK-NOT: distributed.DistributedKernel
// CHECK-NOT: enzymexla_reshard_point

// LOWER: distributed.CastGlobalToLocal %arg0 {{.*}} -> tensor<4x2xf32>
// LOWER-NEXT: stablehlo.exponential
// LOWER-NEXT: stablehlo.negate
// LOWER-NEXT: stablehlo.add
// LOWER-NEXT: distributed.CastLocalToGlobal

// SEARCH: "distributed.DistributedFunction"
// SEARCH: stablehlo.add
// SEARCH-NOT: distributed.Collective
