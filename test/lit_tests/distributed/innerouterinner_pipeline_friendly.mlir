// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards=enable-full-version=true --sdy-reshard-to-collectives --insert-identity-reshard --shardy-to-distributed --localize-distributed-module --cse --distributed-simplify-collectives --distributed-overlap-communication-module %s > /dev/null
// A dot-only pipeline-friendly test case.

sdy.mesh @mesh = <["x"=2]>

distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @innerouterinner_pipeline_friendly(
    %x: tensor<2048x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %w1: tensor<4096x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %w2: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
  %w3: tensor<64x512x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}
) -> (tensor<2048x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %x, %w1 : (tensor<2048x4096xf32>, tensor<4096x64xf32>) -> tensor<2048x64xf32>
  %1 = stablehlo.dot_general %0, %w2, contracting_dims = [] x [] : (tensor<2048x64xf32>, tensor<512xf32>) -> tensor<2048x64x512xf32>
  %2 = stablehlo.dot_general %1, %w3, contracting_dims = [1, 2] x [0, 1] : (tensor<2048x64x512xf32>, tensor<64x512x256xf32>) -> tensor<2048x256xf32>
  return %2 : tensor<2048x256xf32>
}
