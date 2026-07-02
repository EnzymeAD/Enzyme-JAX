// RUN: enzymexlamlir-opt --split-input-file %s --sdy-propagation-pipeline --sdy-resolve-permutation-factors --sdy-export-pipeline="enable-insert-explicit-collectives=true"
// note this currently seems to be the wrong pass pipeline since it produces code without halo
// exchange for convolution.
module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4]>

    func.func @conv_spatial_shard_2d_zero_pad(
        %image: tensor<256x512x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>},
        %kernel: tensor<3x3x1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}
    ) -> (tensor<256x512x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>}) {
        %image_with_batch = stablehlo.reshape %image : (tensor<256x512x1xf32>) -> tensor<1x256x512x1xf32>
        %conv = stablehlo.convolution(%image_with_batch, %kernel)
            dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
            window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]}
            {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
             sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {"b"}, {}]>]>}
            : (tensor<1x256x512x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x256x512x1xf32>
        %conv_no_batch = stablehlo.reshape %conv : (tensor<1x256x512x1xf32>) -> tensor<256x512x1xf32>
        return %conv_no_batch : tensor<256x512x1xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4]>

    func.func @conv_spatial_shard_2d_zero_pad_batched(
        %image: tensor<1x256x512x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>},
        %kernel: tensor<3x3x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}
    ) -> (tensor<1x256x512x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>}) {
        %conv = stablehlo.convolution(%image, %kernel)
            dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
            window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]}
            {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
             sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {"b"}, {}]>]>}
            : (tensor<1x256x512x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x256x512x8xf32>
        return %conv : tensor<1x256x512x8xf32>
    }
}