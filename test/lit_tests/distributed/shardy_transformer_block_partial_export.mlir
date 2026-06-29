module @shardy_transformer_block_pre_export {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]> {stablehlo.mesh = {axes = [{name = "data", size = 4 : i64}, {name = "tile", size = 4 : i64}, {name = "model", size = 2 : i64}]}}
  func.func @transformer_block(%arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>}, %arg1: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg2: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg3: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg4: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg5: tensor<512x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg6: tensor<2048x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"tile"}]>}, %arg7: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}, %arg8: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}) -> (tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
    %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
    %1 = sdy.constant dense<9.99999974E-6> : tensor<f32>
    %2 = sdy.constant dense<1.250000e-01> : tensor<f32>
    %3 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %4 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %5 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %6 = stablehlo.reshape %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %7 = stablehlo.reshape %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %8 = stablehlo.reshape %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %9 = stablehlo.transpose %7, dims = [0, 1, 3, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16x64xf32>) -> tensor<8x8x64x16xf32>
    %10 = stablehlo.dot_general %6, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16x64xf32>, tensor<8x8x64x16xf32>) -> tensor<8x8x16x16xf32>
    %11 = stablehlo.broadcast_in_dim %2, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<f32>) -> tensor<8x8x16x16xf32>
    %12 = stablehlo.multiply %10, %11 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<8x8x16x16xf32>
    %13 = stablehlo.exponential %12 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<8x8x16x16xf32>
    %14 = stablehlo.reduce(%13 init: %0) applies stablehlo.add across dimensions = [3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}]>]>} : (tensor<8x8x16x16xf32>, tensor<f32>) -> tensor<8x8x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16xf32>) -> tensor<8x8x16x16xf32>
    %16 = stablehlo.divide %13, %15 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<8x8x16x16xf32>
    %17 = stablehlo.dot_general %16, %8, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16x16xf32>, tensor<8x8x16x64xf32>) -> tensor<8x8x16x64xf32>
    %18 = stablehlo.reshape %17 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<8x8x16x64xf32>) -> tensor<128x512xf32>
    %19 = stablehlo.dot_general %18, %arg4, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %20 = sdy.reshard %19 <@mesh, [{"data"}, {"tile"}]> : tensor<128x512xf32>
    %21 = stablehlo.add %arg0, %20 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %22 = stablehlo.reduce(%21 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<128x512xf32>, tensor<f32>) -> tensor<128xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128xf32>) -> tensor<128x512xf32>
    %24 = stablehlo.subtract %21, %23 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %25 = stablehlo.multiply %24, %24 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %26 = stablehlo.reduce(%25 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<128x512xf32>, tensor<f32>) -> tensor<128xf32>
    %27 = stablehlo.broadcast_in_dim %1, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<f32>) -> tensor<128xf32>
    %28 = stablehlo.add %26, %27 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : tensor<128xf32>
    %29 = stablehlo.rsqrt %28 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128xf32>) -> tensor<128x512xf32>
    %31 = stablehlo.multiply %24, %30 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %32 = stablehlo.broadcast_in_dim %arg7, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<512xf32>) -> tensor<128x512xf32>
    %33 = stablehlo.broadcast_in_dim %arg8, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<512xf32>) -> tensor<128x512xf32>
    %34 = stablehlo.multiply %31, %32 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %35 = stablehlo.add %34, %33 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %36 = stablehlo.dot_general %35, %arg5, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x2048xf32>) -> tensor<128x2048xf32>
    %37 = stablehlo.broadcast_in_dim %0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<f32>) -> tensor<128x2048xf32>
    %38 = stablehlo.maximum %36, %37 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : tensor<128x2048xf32>
    %39 = stablehlo.dot_general %38, %arg6, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128x2048xf32>, tensor<2048x512xf32>) -> tensor<128x512xf32>
    %40 = stablehlo.add %35, %39 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : tensor<128x512xf32>
    return %40 : tensor<128x512xf32>
  }
}

