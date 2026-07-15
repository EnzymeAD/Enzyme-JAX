// RUN: enzymexlamlir-opt --split-input-file %s --sdy-propagation-pipeline --sdy-export-pipeline="enable-insert-explicit-collectives=true"
module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4, "c" = 2]>

    // test multi axis and where reduction axis isn't its own shard
    // induces: all-reduce
    func.func @dot_ideal(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
        %arg1: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"c"}]>}
    ) -> (tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"c"}]>}) {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"c"}]>]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
        return %0 : tensor<128x512xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4, "c" = 2]>

    // test multi axis and where reduction axis isn't its own shard
    // induces: reduce-scatter
    func.func @dot_shard_transpose(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>},
        %arg1: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>}
    ) -> (tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>}) {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b", "c"}]>]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
        return %0 : tensor<128x512xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4, "c" = 2]>

    // Complex example inducing back-to-back collectives including
    // all-slice, collective-permute, reduce-scatter, and all-gather.
    func.func @dot_incompatible_shards(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
        %arg1: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}
    ) -> (tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {"b"}]>}) {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b", "c"}]>]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
        return %0 : tensor<128x512xf32>
    }
}