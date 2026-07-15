// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards=enable-full-version=true --sdy-reshard-to-collectives --insert-identity-reshard --shardy-to-distributed --localize-distributed-module --cse --distributed-simplify-collectives --distributed-overlap-communication-module %s > /dev/null
// A branched, dot-only pipeline-friendly test case.
//
// Shape:
//   f(g(a, b), h(c, d))
//
// Design intent:
// 1) Two independent, uneven branches create useful local work before the
//    merge.
// 2) Avoid reduction collectives in the computation.
// 3) Force gather-style movement on the right branch at the final merge by
//    sharding a contracting dimension of that branch's result.

sdy.mesh @mesh = <["x"=2]>

distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @branched_pipeline_friendly(
    %a: tensor<2048x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %b: tensor<4096x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %c: tensor<64x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %d: tensor<1024x128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>},
    %u: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}
) -> (tensor<2048x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %a, %b : (tensor<2048x4096xf32>, tensor<4096x64xf32>) -> tensor<2048x64xf32>
  %1 = stablehlo.dot_general %0, %u, contracting_dims = [] x [] : (tensor<2048x64xf32>, tensor<128xf32>) -> tensor<2048x64x128xf32>

  %2 = stablehlo.dot_general %c, %d, contracting_dims = [1] x [0] : (tensor<64x1024xf32>, tensor<1024x128x256xf32>) -> tensor<64x128x256xf32>

  %3 = stablehlo.dot_general %1, %2, contracting_dims = [1, 2] x [0, 1] : (tensor<2048x64x128xf32>, tensor<64x128x256xf32>) -> tensor<2048x256xf32>
  return %3 : tensor<2048x256xf32>
}