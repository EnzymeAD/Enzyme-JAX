// RUN: enzymexlamlir-opt --sdy-propagation-pipeline '--sdy-insert-explicit-reshards=enable-full-version=true' --sdy-reshard-to-collectives %s | FileCheck %s
// A somewhat unrealistic example of an inner-product * outer-product * inner-product which should
// be ideal for pipelining, on the basis that only communicating the inner products is much cheaper
// than communicating the outer products.
// (not entirely unrealistic- maybe there is a low-rank approximation being used).

sdy.mesh @mesh = <["x"=2]>

distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @innerouterinnerdataparallel(
    %x: tensor<512x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %w1: tensor<1024x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %w2: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
    %w3: tensor<64x2x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}
) -> (tensor<512x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
    %0 = stablehlo.dot %x, %w1 : (tensor<512x1024xf32>, tensor<1024x64xf32>) -> tensor<512x64xf32>
    %1 = stablehlo.dot_general %0, %w2, contracting_dims = [] x [] : (tensor<512x64xf32>, tensor<2xf32>) -> tensor<512x64x2xf32>
    %2 = stablehlo.dot_general %1, %w3, contracting_dims = [1, 2] x [0, 1] : (tensor<512x64x2xf32>, tensor<64x2x64xf32>) -> tensor<512x64xf32>
    return %2 : tensor<512x64xf32>
}