// RUN: enzymexlamlir-opt --shardy-to-distributed-pipeline %s | FileCheck %s
// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @m device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s

// A reshard feeding the return. The function result is delivered in the
// reshard's output axes, so erasing the reshard must leave a collective between
// the negate and the return, in both search and Shardy baseline mode.

module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(%x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
    %0 = stablehlo.negate %x {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
    %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }
}

// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.negate
// CHECK: %[[OUT:.*]] = distributed.CastLocalToGlobal
// CHECK: %[[LOCAL:.*]] = distributed.CastGlobalToLocal %[[OUT]]
// CHECK: %[[C:.*]] = distributed.Collective %[[LOCAL]] {{.*}} reduces ()
// CHECK: %[[AWAIT:.*]] = distributed.Await %[[C]]
// CHECK: %[[RES:.*]] = distributed.CastLocalToGlobal %[[AWAIT]]
// CHECK: distributed.DistributedYield (%[[RES]] : tensor<8x8xf32>)
