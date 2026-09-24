// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true --mlir-print-ir-after=cluster-distributed-kernels %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s --check-prefix=FINAL
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline --mlir-print-ir-after=cluster-distributed-kernels %s 2>&1 | FileCheck %s --check-prefix=OFF

// With the flag, the sdy.mesh and the per-op sdy.sharding survive kernel
// clustering, and the mesh is dropped only afterwards. Without it the mesh is
// gone by then.

module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
    %0 = stablehlo.negate %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x8xf32>
    %1 = sdy.reshard %0 <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
    %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : tensor<8x8xf32>
    return %2 : tensor<8x8xf32>
  }
}

// ON: IR Dump After ClusterDistributedKernelsPass
// ON: sdy.mesh @mesh = <["a"=2, "b"=2]>
// ON: distributed.DistributedFunction
// ON-SAME: arg_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}]
// ON-SAME: res_attrs = [{sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}]
// ON: stablehlo.negate {{.*}}sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
// ON: stablehlo.negate {{.*}}sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>

// FINAL-NOT: sdy.mesh
// FINAL: distributed.DistributedFunction

// OFF: IR Dump After ClusterDistributedKernelsPass
// OFF-NOT: sdy.mesh
// OFF: distributed.DistributedFunction
