// RUN: enzymexlamlir-opt --sdy-propagation-pipeline  --sdy-insert-explicit-reshards=enable-full-version=true --sdy-reshard-to-collectives --shardy-to-distributed --localize-distributed-module --cse | FileCheck %s
// A somewhat unrealistic example of an inner-product * outer-product * inner-product which should
// be ideal for pipelining, on the basis that only communicating the inner products is much cheaper
// than communicating the outer products.
// (not entirely unrealistic- maybe there is a low-rank approximation being used).

sdy.mesh @mesh = <["x"=2]>
distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @innerouterinneroperatorparallel(
    %x: tensor<512x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
    %w1: tensor<1024x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %w2: tensor<1x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %w3: tensor<8192x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}
) -> (tensor<512x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
    %0 = stablehlo.dot %x, %w1 : (tensor<512x1024xf32>, tensor<1024x64xf32>) -> tensor<512x64xf32>
    %1 = stablehlo.reshape %0 : (tensor<512x64xf32>) -> tensor<32768x1xf32>
    %2 = stablehlo.dot %1, %w2 : (tensor<32768x1xf32>, tensor<1x128xf32>) -> tensor<32768x128xf32>
    %3 = stablehlo.reshape %2 : (tensor<32768x128xf32>) -> tensor<512x8192xf32>
    %4 = stablehlo.dot %3, %w3 : (tensor<512x8192xf32>, tensor<8192x10xf32>) -> tensor<512x10xf32>
    return %4 : tensor<512x10xf32>
}

// TODO: filecheck expected output