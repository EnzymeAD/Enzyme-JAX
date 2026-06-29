module @shardy_transformer_block_pre_export {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]> {stablehlo.mesh = {axes = [{name = "data", size = 4 : i64}, {name = "tile", size = 4 : i64}, {name = "model", size = 2 : i64}]}}
  func.func @transformer_block(%arg0: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>}, %arg1: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg2: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg3: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg4: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg5: tensor<128x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>}, %arg6: tensor<1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"tile"}]>}, %arg7: tensor<256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}, %arg8: tensor<256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>}) -> (tensor<32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<32x128xf32>, tensor<128x256xf32>) -> tensor<32x256xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32x256xf32>) -> tensor<32x256xf32>
    %2 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<32x128xf32>, tensor<128x256xf32>) -> tensor<32x256xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32x256xf32>) -> tensor<32x256xf32>
    %4 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<32x128xf32>, tensor<128x256xf32>) -> tensor<32x256xf32>
    %5 = "stablehlo.all_reduce"(%4) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32x256xf32>) -> tensor<32x256xf32>
    %6 = "stablehlo.all_gather"(%1) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>, use_global_device_ids}> : (tensor<32x256xf32>) -> tensor<32x512xf32>
    %7 = stablehlo.reshape %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<32x512xf32>) -> tensor<2x8x16x64xf32>
    %8 = "stablehlo.all_gather"(%3) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>, use_global_device_ids}> : (tensor<32x256xf32>) -> tensor<32x512xf32>
    %9 = stablehlo.reshape %8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<32x512xf32>) -> tensor<2x8x16x64xf32>
    %10 = "stablehlo.all_gather"(%5) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>, use_global_device_ids}> : (tensor<32x256xf32>) -> tensor<32x512xf32>
    %11 = stablehlo.reshape %10 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<32x512xf32>) -> tensor<2x8x16x64xf32>
    %12 = stablehlo.transpose %9, dims = [0, 1, 3, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<2x8x16x64xf32>) -> tensor<2x8x64x16xf32>
    %13 = stablehlo.dot_general %7, %12, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<2x8x16x64xf32>, tensor<2x8x64x16xf32>) -> tensor<2x8x16x16xf32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<f32>) -> tensor<2x8x16x16xf32>
    %15 = stablehlo.multiply %13, %14 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<2x8x16x16xf32>
    %16 = stablehlo.exponential %15 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<2x8x16x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}]>]>} : (tensor<2x8x16x16xf32>, tensor<f32>) -> tensor<2x8x16xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<2x8x16xf32>) -> tensor<2x8x16x16xf32>
    %19 = stablehlo.divide %16, %18 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : tensor<2x8x16x16xf32>
    %20 = stablehlo.dot_general %19, %11, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}, {}, {}]>]>} : (tensor<2x8x16x16xf32>, tensor<2x8x16x64xf32>) -> tensor<2x8x16x64xf32>
    %21 = stablehlo.reshape %20 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {}]>]>} : (tensor<2x8x16x64xf32>) -> tensor<32x512xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %22 = stablehlo.partition_id : tensor<ui32>
    %23 = stablehlo.convert %22 : (tensor<ui32>) -> tensor<i64>
    %c_2 = stablehlo.constant dense<[0, 0, 128, 128, 256, 256, 384, 384, 0, 0, 128, 128, 256, 256, 384, 384, 0, 0, 128, 128, 256, 256, 384, 384, 0, 0, 128, 128, 256, 256, 384, 384]> : tensor<32xi64>
    %24 = stablehlo.dynamic_slice %c_2, %23, sizes = [1] : (tensor<32xi64>, tensor<i64>) -> tensor<1xi64>
    %25 = stablehlo.reshape %24 : (tensor<1xi64>) -> tensor<i64>
    %26 = stablehlo.dynamic_slice %21, %c, %25, sizes = [32, 128] : (tensor<32x512xf32>, tensor<i64>, tensor<i64>) -> tensor<32x128xf32>
    %27 = stablehlo.dot_general %26, %arg4, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<32x128xf32>, tensor<128x256xf32>) -> tensor<32x256xf32>
    %28 = "stablehlo.all_reduce"(%27) <{channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32x256xf32>) -> tensor<32x256xf32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %29 = stablehlo.partition_id : tensor<ui32>
    %30 = stablehlo.convert %29 : (tensor<ui32>) -> tensor<i64>
    %c_4 = stablehlo.constant dense<[0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128]> : tensor<32xi64>
    %31 = stablehlo.dynamic_slice %c_4, %30, sizes = [1] : (tensor<32xi64>, tensor<i64>) -> tensor<1xi64>
    %32 = stablehlo.reshape %31 : (tensor<1xi64>) -> tensor<i64>
    %33 = stablehlo.dynamic_slice %28, %c_3, %32, sizes = [32, 128] : (tensor<32x256xf32>, tensor<i64>, tensor<i64>) -> tensor<32x128xf32>
    %34 = "stablehlo.collective_permute"(%33) <{channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, source_target_pairs = dense<[[0, 0], [1, 2], [2, 4], [3, 6], [4, 1], [5, 3], [6, 5], [7, 7], [8, 8], [9, 10], [10, 12], [11, 14], [12, 9], [13, 11], [14, 13], [15, 15], [16, 16], [17, 18], [18, 20], [19, 22], [20, 17], [21, 19], [22, 21], [23, 23], [24, 24], [25, 26], [26, 28], [27, 30], [28, 25], [29, 27], [30, 29], [31, 31]]> : tensor<32x2xi64>}> : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %35 = stablehlo.add %arg0, %34 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %36 = stablehlo.reduce(%35 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<32x128xf32>, tensor<f32>) -> tensor<32xf32>
    %37 = "stablehlo.all_reduce"(%36) <{channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32xf32>) -> tensor<32xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<32xf32>) -> tensor<32x128xf32>
    %39 = stablehlo.subtract %35, %38 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %40 = stablehlo.multiply %39, %39 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %41 = stablehlo.reduce(%40 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<32x128xf32>, tensor<f32>) -> tensor<32xf32>
    %42 = "stablehlo.all_reduce"(%41) <{channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32xf32>) -> tensor<32xf32>
    %43 = stablehlo.broadcast_in_dim %cst_0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : (tensor<f32>) -> tensor<32xf32>
    %44 = stablehlo.add %42, %43 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : tensor<32xf32>
    %45 = stablehlo.rsqrt %44 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}]>]>} : tensor<32xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<32xf32>) -> tensor<32x128xf32>
    %47 = stablehlo.multiply %39, %46 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %48 = stablehlo.partition_id : tensor<ui32>
    %49 = stablehlo.convert %48 : (tensor<ui32>) -> tensor<i64>
    %c_5 = stablehlo.constant dense<[0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128]> : tensor<32xi64>
    %50 = stablehlo.dynamic_slice %c_5, %49, sizes = [1] : (tensor<32xi64>, tensor<i64>) -> tensor<1xi64>
    %51 = stablehlo.reshape %50 : (tensor<1xi64>) -> tensor<i64>
    %52 = stablehlo.dynamic_slice %arg7, %51, sizes = [128] : (tensor<256xf32>, tensor<i64>) -> tensor<128xf32>
    %53 = "stablehlo.collective_permute"(%52) <{channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, source_target_pairs = dense<[[0, 0], [1, 2], [2, 4], [3, 6], [4, 1], [5, 3], [6, 5], [7, 7], [8, 8], [9, 10], [10, 12], [11, 14], [12, 9], [13, 11], [14, 13], [15, 15], [16, 16], [17, 18], [18, 20], [19, 22], [20, 17], [21, 19], [22, 21], [23, 23], [24, 24], [25, 26], [26, 28], [27, 30], [28, 25], [29, 27], [30, 29], [31, 31]]> : tensor<32x2xi64>}> : (tensor<128xf32>) -> tensor<128xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128xf32>) -> tensor<32x128xf32>
    %55 = stablehlo.partition_id : tensor<ui32>
    %56 = stablehlo.convert %55 : (tensor<ui32>) -> tensor<i64>
    %c_6 = stablehlo.constant dense<[0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128]> : tensor<32xi64>
    %57 = stablehlo.dynamic_slice %c_6, %56, sizes = [1] : (tensor<32xi64>, tensor<i64>) -> tensor<1xi64>
    %58 = stablehlo.reshape %57 : (tensor<1xi64>) -> tensor<i64>
    %59 = stablehlo.dynamic_slice %arg8, %58, sizes = [128] : (tensor<256xf32>, tensor<i64>) -> tensor<128xf32>
    %60 = "stablehlo.collective_permute"(%59) <{channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, source_target_pairs = dense<[[0, 0], [1, 2], [2, 4], [3, 6], [4, 1], [5, 3], [6, 5], [7, 7], [8, 8], [9, 10], [10, 12], [11, 14], [12, 9], [13, 11], [14, 13], [15, 15], [16, 16], [17, 18], [18, 20], [19, 22], [20, 17], [21, 19], [22, 21], [23, 23], [24, 24], [25, 26], [26, 28], [27, 30], [28, 25], [29, 27], [30, 29], [31, 31]]> : tensor<32x2xi64>}> : (tensor<128xf32>) -> tensor<128xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<128xf32>) -> tensor<32x128xf32>
    %62 = stablehlo.multiply %47, %54 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %63 = stablehlo.add %62, %61 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %64 = stablehlo.dot_general %63, %arg5, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<32x128xf32>, tensor<128x1024xf32>) -> tensor<32x1024xf32>
    %65 = "stablehlo.all_reduce"(%64) <{channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23], [24, 26, 28, 30], [25, 27, 29, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %66 = stablehlo.broadcast_in_dim %cst, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : (tensor<f32>) -> tensor<32x1024xf32>
    %67 = stablehlo.maximum %65, %66 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"model"}]>]>} : tensor<32x1024xf32>
    %68 = stablehlo.dot_general %67, %arg6, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : (tensor<32x1024xf32>, tensor<1024x128xf32>) -> tensor<32x128xf32>
    %69 = "stablehlo.all_reduce"(%68) <{channel_handle = #stablehlo.channel_handle<handle = 14, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %73 = stablehlo.add %arg9, %arg10 : tensor<f32>
      stablehlo.return %73 : tensor<f32>
    }) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %70 = stablehlo.add %63, %69 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"data"}, {"tile"}]>]>} : tensor<32x128xf32>
    %71 = "stablehlo.collective_permute"(%70) <{channel_handle = #stablehlo.channel_handle<handle = 15, type = 1>, source_target_pairs = dense<[[0, 0], [1, 2], [2, 1], [3, 3], [4, 4], [5, 6], [6, 5], [7, 7], [8, 8], [9, 10], [10, 9], [11, 11], [12, 12], [13, 14], [14, 13], [15, 15], [16, 16], [17, 18], [18, 17], [19, 19], [20, 20], [21, 22], [22, 21], [23, 23], [24, 24], [25, 26], [26, 25], [27, 27], [28, 28], [29, 30], [30, 29], [31, 31]]> : tensor<32x2xi64>}> : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %72 = "stablehlo.all_gather"(%71) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 16, type = 1>, replica_groups = dense<[[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15], [16, 18], [17, 19], [20, 22], [21, 23], [24, 26], [25, 27], [28, 30], [29, 31]]> : tensor<16x2xi64>, use_global_device_ids}> : (tensor<32x128xf32>) -> tensor<32x256xf32>
    return %72 : tensor<32x256xf32>
  }
}

