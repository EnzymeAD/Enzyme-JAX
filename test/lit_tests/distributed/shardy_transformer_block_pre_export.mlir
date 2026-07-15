// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-export-pipeline="enable-insert-explicit-collectives=true" --sdy-convert-global-to-local="enable-rgv3=true" -o /dev/null %s
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-export-pipeline="enable-insert-explicit-collectives=true" --sdy-convert-global-to-local="enable-rgv3=true" --insert-physical-mesh --shardy-to-distributed -o /dev/null %s

module @shardy_transformer_block_pre_export {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]>

  func.func @transformer_block(
      %arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>},
      %arg1: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg2: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg3: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg4: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg5: tensor<512x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg6: tensor<2048x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"tile"}]>},
      %arg7: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>},
      %arg8: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>})
      -> (tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %ceps = stablehlo.constant dense<1.000000e-05> : tensor<f32>
    %cscale = stablehlo.constant dense<1.250000e-01> : tensor<f32>

    // QKV projections.
    %q = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %k = stablehlo.dot_general %arg0, %arg2, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v = stablehlo.dot_general %arg0, %arg3, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>

    // Multi-head attention as [batch=8, heads=8, seq=16, head_dim=64].
    %q_heads = stablehlo.reshape %q : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %k_heads = stablehlo.reshape %k : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %v_heads = stablehlo.reshape %v : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>

    %k_t = stablehlo.transpose %k_heads, dims = [0, 1, 3, 2] :
      (tensor<8x8x16x64xf32>) -> tensor<8x8x64x16xf32>
    %scores = stablehlo.dot_general %q_heads, %k_t, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] :
      (tensor<8x8x16x64xf32>, tensor<8x8x64x16xf32>) -> tensor<8x8x16x16xf32>

    %scale_bcast = stablehlo.broadcast_in_dim %cscale, dims = [] :
      (tensor<f32>) -> tensor<8x8x16x16xf32>
    %scores_scaled = stablehlo.multiply %scores, %scale_bcast : tensor<8x8x16x16xf32>

    // Softmax-like normalization using exp/reduce/divide.
    %weights = stablehlo.exponential %scores_scaled : tensor<8x8x16x16xf32>
    %weights_sum = stablehlo.reduce(%weights init: %c0) applies stablehlo.add across dimensions = [3] :
      (tensor<8x8x16x16xf32>, tensor<f32>) -> tensor<8x8x16xf32>
    %weights_sum_bcast = stablehlo.broadcast_in_dim %weights_sum, dims = [0, 1, 2] :
      (tensor<8x8x16xf32>) -> tensor<8x8x16x16xf32>
    %weights_norm = stablehlo.divide %weights, %weights_sum_bcast : tensor<8x8x16x16xf32>

    %ctx_heads = stablehlo.dot_general %weights_norm, %v_heads, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] :
      (tensor<8x8x16x16xf32>, tensor<8x8x16x64xf32>) -> tensor<8x8x16x64xf32>
    %ctx = stablehlo.reshape %ctx_heads : (tensor<8x8x16x64xf32>) -> tensor<128x512xf32>

    // Output projection + residual.
    %attn_out = stablehlo.dot_general %ctx, %arg4, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %resid0 = stablehlo.add %arg0, %attn_out : tensor<128x512xf32>

    // LayerNorm-like normalization.
    %mean = stablehlo.reduce(%resid0 init: %c0) applies stablehlo.add across dimensions = [1] :
      (tensor<128x512xf32>, tensor<f32>) -> tensor<128xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0] :
      (tensor<128xf32>) -> tensor<128x512xf32>
    %centered = stablehlo.subtract %resid0, %mean_bcast : tensor<128x512xf32>
    %sq = stablehlo.multiply %centered, %centered : tensor<128x512xf32>
    %var = stablehlo.reduce(%sq init: %c0) applies stablehlo.add across dimensions = [1] :
      (tensor<128x512xf32>, tensor<f32>) -> tensor<128xf32>
    %eps_bcast = stablehlo.broadcast_in_dim %ceps, dims = [] :
      (tensor<f32>) -> tensor<128xf32>
    %var_eps = stablehlo.add %var, %eps_bcast : tensor<128xf32>
    %inv_std = stablehlo.rsqrt %var_eps : tensor<128xf32>
    %inv_std_bcast = stablehlo.broadcast_in_dim %inv_std, dims = [0] :
      (tensor<128xf32>) -> tensor<128x512xf32>
    %norm = stablehlo.multiply %centered, %inv_std_bcast : tensor<128x512xf32>

    %gamma = stablehlo.broadcast_in_dim %arg7, dims = [1] :
      (tensor<512xf32>) -> tensor<128x512xf32>
    %beta = stablehlo.broadcast_in_dim %arg8, dims = [1] :
      (tensor<512xf32>) -> tensor<128x512xf32>
    %norm_scaled = stablehlo.multiply %norm, %gamma : tensor<128x512xf32>
    %norm_out = stablehlo.add %norm_scaled, %beta : tensor<128x512xf32>

    // MLP block + residual.
    %ff1 = stablehlo.dot_general %norm_out, %arg5, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x2048xf32>) -> tensor<128x2048xf32>
    %zero_ff = stablehlo.broadcast_in_dim %c0, dims = [] :
      (tensor<f32>) -> tensor<128x2048xf32>
    %ff1_relu = stablehlo.maximum %ff1, %zero_ff : tensor<128x2048xf32>
    %ff2 = stablehlo.dot_general %ff1_relu, %arg6, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x2048xf32>, tensor<2048x512xf32>) -> tensor<128x512xf32>
    %out = stablehlo.add %norm_out, %ff2 : tensor<128x512xf32>

    return %out : tensor<128x512xf32>
  }
}
