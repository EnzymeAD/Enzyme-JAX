// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline=dump-value-axes=true %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DUMP
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s --check-prefix=BASE
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | enzymexlamlir-opt --distributed-search-lowering-pipeline="finalize-schedule=false" 2>/dev/null | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e6, device_mem_bandwidth = 1.0e9}"' --distributed-search-strategies="beam-size=8 fail-on-bad-candidate" %s 2>/dev/null | FileCheck %s --check-prefix=SEARCH

// A reshard point on the key sequence ahead of a grouped-query KV repeat.
// Its rule breaks dim 1 (factor j on the operand, l on the result) and passes
// the kv heads (i) and head dim (k) through, so a collective may change only
// the key sequence's sharding here. The sharding constraint after the break
// replicates that dimension in the Shardy plan, so the baseline gathers the
// keys at the break rather than at the dot.

module {
  sdy.mesh @mesh = <["data"=2, "model"=2]>
  func.func @main(%q: tensor<4x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}, {}]>}, %k: tensor<2x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}, {}]>}) -> (tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}, {}]>}) {
    %b = stablehlo.custom_call @enzymexla_reshard_point(%k) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, k]) {i=2, j=8, k=4, l=8}, custom>} : (tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
    %c = sdy.sharding_constraint %b <@mesh, [{?}, {}, {?}]> : tensor<2x8x4xf32>
    %r = stablehlo.broadcast_in_dim %c, dims = [0, 2, 3] : (tensor<2x8x4xf32>) -> tensor<2x2x8x4xf32>
    %rr = stablehlo.reshape %r : (tensor<2x2x8x4xf32>) -> tensor<4x8x4xf32>
    %s = stablehlo.dot_general %q, %rr, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<4x8x4xf32>, tensor<4x8x4xf32>) -> tensor<4x8x8xf32>
    return %s : tensor<4x8x8xf32>
  }
}

// The heads and head dim of %k keep their axes across the break; its key
// sequence takes a new one, which the repeat and the dot then carry.
// DUMP: ===== Logical Axes per Value =====
// DUMP: %arg1 : {{\[}}[[H:%[0-9]+]]{{\]}} {{\[}}[[T:%[0-9]+]]{{\]}} {{\[}}[[D:%[0-9]+]]{{\]}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[H]]{{\]}} {{\[}}[[TB:a[0-9]+]]{{\]}} {{\[}}[[D]]{{\]}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[H]]{{\]}} {{\[}}[[TB]]{{\]}} {{\[}}[[D]]{{\]}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[H]]{{\]}} {{\[}}%{{[0-9]+}}{{\]}} {{\[}}[[TB]]{{\]}} {{\[}}[[D]]{{\]}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[H]], %{{[0-9]+}}{{\]}} {{\[}}[[TB]]{{\]}} {{\[}}[[D]]{{\]}}
// DUMP: ===== End Logical Axes per Value =====

// Search mode: the break is gone and a collective between the two key
// sequence axes takes its place, directly on the argument.
// CHECK: ^bb0(%{{.*}}: tensor<4x8x4xf32>, %[[K:.*]]: tensor<2x8x4xf32>):
// CHECK-NEXT: %[[KL:.*]] = distributed.CastGlobalToLocal %[[K]]
// CHECK-NEXT: distributed.Collective %[[KL]] : tensor<1x1x1xf32> on %{{.*}} : <64> to tensor<1x1x1xf32> on %{{.*}} : <64> reduces ()
// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.broadcast_in_dim
// CHECK-NOT: enzymexla_reshard_point

// BASE: ^bb0(%{{.*}}: tensor<4x8x4xf32>, %[[K:.*]]: tensor<2x8x4xf32>):
// BASE-NEXT: %[[KL:.*]] = distributed.CastGlobalToLocal %[[K]]
// BASE-NEXT: distributed.Collective %[[KL]] : tensor<1x1x1xf32> on %{{.*}} : <64> to tensor<1x1x1xf32> on %{{.*}} : <64> reduces ()
// BASE-NOT: enzymexla_reshard_point

// The keys arrive split over "data" (4 of 8 positions per device) and are
// gathered to all 8 before the repeat; the dot itself needs no collective.
// LOWER: %[[KL:.*]] = distributed.CastGlobalToLocal %arg1 {{.*}} -> tensor<1x4x4xf32>
// LOWER: stablehlo.all_gather
// LOWER: %[[KG:.*]] = stablehlo.reshape %{{.*}} : (tensor<32xf32>) -> tensor<1x8x4xf32>
// LOWER-NEXT: stablehlo.broadcast_in_dim %[[KG]], dims = [0, 2, 3] {{.*}}: (tensor<1x8x4xf32>) -> tensor<1x2x8x4xf32>
// LOWER-NOT: all_gather
// LOWER-NOT: all_reduce
// LOWER: distributed.DistributedYield

// SEARCH: "distributed.DistributedFunction"
// SEARCH: stablehlo.dot_general
// SEARCH-NOT: enzymexla_reshard_point
