// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards=enable-full-version=true --sdy-reshard-to-collectives --insert-identity-reshard --shardy-to-distributed --localize-distributed-module --cse --distributed-simplify-collectives --distributed-overlap-communication-module %s > /dev/null
// A diamond-shaped, dot-only pipeline-friendly test case.
//
// Shape:
//   a1 = a0 * x
//   b1 = b0 * a1
//   a2 = a0 * y
//   b2 = b1 * a2
//
// Design intent:
// 1) a0 feeds two branches, creating a true dependency diamond.
// 2) The two branches do different work / use different output sizes.
// 3) Avoid all-reduce and prefer gather-style communication on sharded inputs.

sdy.mesh @mesh = <["x"=2]>

distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @diamond_pipeline_friendly(
    %a0: tensor<1024x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %b0: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %x: tensor<2048x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %y: tensor<2048x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}
) -> (tensor<64x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.dot %a0, %x : (tensor<1024x2048xf32>, tensor<2048x64xf32>) -> tensor<1024x64xf32>
  %1 = stablehlo.dot_general %b0, %0, contracting_dims = [1] x [0] : (tensor<1024x1024xf32>, tensor<1024x64xf32>) -> tensor<1024x64xf32>
  %2 = stablehlo.dot %a0, %y : (tensor<1024x2048xf32>, tensor<2048x96xf32>) -> tensor<1024x96xf32>
  %3 = stablehlo.dot_general %1, %2, contracting_dims = [0] x [0] : (tensor<1024x64xf32>, tensor<1024x96xf32>) -> tensor<64x96xf32>
  return %3 : tensor<64x96xf32>
}
