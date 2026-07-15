module @shardy_transformer_block_pre_export {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]> {stablehlo.mesh = {axes = [{name = "data", size = 4 : i64}, {name = "tile", size = 4 : i64}, {name = "model", size = 2 : i64}]}}
  func.func @transformer_block(%arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>}, %arg1: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg2: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg3: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg4: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg5: tensor<512x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg6: tensor<2048x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"tile"}]>}, %arg7: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}, %arg8: tensor<512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}) -> (tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
    %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
    %1 = sdy.constant dense<9.99999974E-6> : tensor<f32>
    %2 = sdy.constant dense<1.250000e-01> : tensor<f32>
    %3 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %4 = sdy.all_reduce {"tile"} %3 out_sharding=<@mesh, [{"data"}, {"model"}]> : tensor<128x512xf32>
    %5 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %6 = sdy.all_reduce {"tile"} %5 out_sharding=<@mesh, [{"data"}, {"model"}]> : tensor<128x512xf32>
    %7 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %8 = sdy.all_reduce {"tile"} %7 out_sharding=<@mesh, [{"data"}, {"model"}]> : tensor<128x512xf32>
    %9 = sdy.all_gather [{}, {"model"}] %4 out_sharding=<@mesh, [{"data"}, {}]> : tensor<128x512xf32>
    %10 = stablehlo.reshape %9 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %11 = sdy.all_gather [{}, {"model"}] %6 out_sharding=<@mesh, [{"data"}, {}]> : tensor<128x512xf32>
    %12 = stablehlo.reshape %11 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %13 = sdy.all_gather [{}, {"model"}] %8 out_sharding=<@mesh, [{"data"}, {}]> : tensor<128x512xf32>
    %14 = stablehlo.reshape %13 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %15 = stablehlo.transpose %12, dims = [0, 1, 3, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16x64xf32>) -> tensor<8x8x64x16xf32>
    %16 = stablehlo.dot_general %10, %15, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16x64xf32>, tensor<8x8x64x16xf32>) -> tensor<8x8x16x16xf32>
    %17 = stablehlo.broadcast_in_dim %2, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<f32>) -> tensor<8x8x16x16xf32>
    %18 = stablehlo.multiply %16, %17 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<8x8x16x16xf32>
    %19 = stablehlo.exponential %18 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<8x8x16x16xf32>
    %20 = stablehlo.reduce(%19 init: %0) applies stablehlo.add across dimensions = [3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}]>]>} : (tensor<8x8x16x16xf32>, tensor<f32>) -> tensor<8x8x16xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16xf32>) -> tensor<8x8x16x16xf32>
    %22 = stablehlo.divide %19, %21 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<8x8x16x16xf32>
    %23 = stablehlo.dot_general %22, %14, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<8x8x16x16xf32>, tensor<8x8x16x64xf32>) -> tensor<8x8x16x64xf32>
    %24 = stablehlo.reshape %23 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}]>]>} : (tensor<8x8x16x64xf32>) -> tensor<128x512xf32>
    %25 = sdy.all_slice [{}, {"tile"}] %24 out_sharding=<@mesh, [{"data"}, {"tile"}]> : tensor<128x512xf32>
    %26 = stablehlo.dot_general %25, %arg4, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %27 = sdy.all_reduce {"tile"} %26 out_sharding=<@mesh, [{"data"}, {"model"}]> : tensor<128x512xf32>
    %28 = sdy.all_slice [{}, {"tile":(1)2}] %27 out_sharding=<@mesh, [{"data"}, {"model", "tile":(1)2}]> : tensor<128x512xf32>
    %29 = sdy.collective_permute %28 out_sharding=<@mesh, [{"data"}, {"tile"}]> : tensor<128x512xf32>
    %30 = stablehlo.add %arg0, %29 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %31 = stablehlo.reduce(%30 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<128x512xf32>, tensor<f32>) -> tensor<128xf32>
    %32 = sdy.all_reduce {"tile"} %31 out_sharding=<@mesh, [{"data"}]> : tensor<128xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128xf32>) -> tensor<128x512xf32>
    %34 = stablehlo.subtract %30, %33 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %35 = stablehlo.multiply %34, %34 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %36 = stablehlo.reduce(%35 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<128x512xf32>, tensor<f32>) -> tensor<128xf32>
    %37 = sdy.all_reduce {"tile"} %36 out_sharding=<@mesh, [{"data"}]> : tensor<128xf32>
    %38 = stablehlo.broadcast_in_dim %1, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<f32>) -> tensor<128xf32>
    %39 = stablehlo.add %37, %38 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : tensor<128xf32>
    %40 = stablehlo.rsqrt %39 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : tensor<128xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128xf32>) -> tensor<128x512xf32>
    %42 = stablehlo.multiply %34, %41 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %43 = sdy.all_slice [{"tile":(1)2}] %arg7 out_sharding=<@mesh, [{"model", "tile":(1)2}]> : tensor<512xf32>
    %44 = sdy.collective_permute %43 out_sharding=<@mesh, [{"tile"}]> : tensor<512xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<512xf32>) -> tensor<128x512xf32>
    %46 = sdy.all_slice [{"tile":(1)2}] %arg8 out_sharding=<@mesh, [{"model", "tile":(1)2}]> : tensor<512xf32>
    %47 = sdy.collective_permute %46 out_sharding=<@mesh, [{"tile"}]> : tensor<512xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<512xf32>) -> tensor<128x512xf32>
    %49 = stablehlo.multiply %42, %45 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %50 = stablehlo.add %49, %48 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %51 = stablehlo.dot_general %50, %arg5, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<128x512xf32>, tensor<512x2048xf32>) -> tensor<128x2048xf32>
    %52 = sdy.all_reduce {"tile"} %51 out_sharding=<@mesh, [{"data"}, {"model"}]> : tensor<128x2048xf32>
    %53 = stablehlo.broadcast_in_dim %0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<f32>) -> tensor<128x2048xf32>
    %54 = stablehlo.maximum %52, %53 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : tensor<128x2048xf32>
    %55 = stablehlo.dot_general %54, %arg6, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128x2048xf32>, tensor<2048x512xf32>) -> tensor<128x512xf32>
    %56 = sdy.all_reduce {"model"} %55 out_sharding=<@mesh, [{"data"}, {"tile"}]> : tensor<128x512xf32>
    %57 = stablehlo.add %50, %56 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<128x512xf32>
    %58 = sdy.collective_permute %57 out_sharding=<@mesh, [{"data"}, {"model", "tile":(2)2}]> : tensor<128x512xf32>
    %59 = sdy.all_gather [{}, {"tile":(2)2}] %58 out_sharding=<@mesh, [{"data"}, {"model"}]> : tensor<128x512xf32>
    return %59 : tensor<128x512xf32>
  }
}

