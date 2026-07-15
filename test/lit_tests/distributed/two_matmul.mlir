// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards=enable-full-version=true --sdy-reshard-to-collectives --insert-identity-reshard --shardy-to-distributed --localize-distributed-module --cse --distributed-simplify-collectives --distributed-overlap-communication-module --distributed-sink-recvs-module %s > /dev/null
// Two chained square matmuls: t1 = A @ B, t2 = t1 @ C.
// A is row-sharded; B and C are replicated so no all-reduce arises.

sdy.mesh @mesh = <["x"=2]>

distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @two_matmul(
    %a: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %b: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %c: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}
) -> (tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %t1 = stablehlo.dot %a, %b : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %t2 = stablehlo.dot %t1, %c : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %t2 : tensor<1024x1024xf32>
}
