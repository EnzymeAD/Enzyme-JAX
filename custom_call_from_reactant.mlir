module @"reactant_#3" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"##_fwd_reverse#281"(%arg0: tensor<1x1x2x2x2xf32>, %arg1: tensor<1x1x2x2x2xf32>, %arg2: tensor<1x1x2x2x2xf32>, %arg3: tensor<1x1x2x2xf32>, %arg4: tensor<1x1x2x2xi1>, %arg5: tensor<1x1x2x2x2xf32>, %arg6: tensor<1x1x2x2xf32>, %arg7: tensor<1x1x2x2x2xf32>, %arg8: tensor<1x1x2x2xf32>) -> (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>, tensor<1x1x2x2xf32>) {
    %0 = stablehlo.multiply %arg7, %arg7 : tensor<1x1x2x2x2xf32>
    %1 = stablehlo.add %arg7, %arg5 : tensor<1x1x2x2x2xf32>
    %2 = stablehlo.add %arg7, %arg7 : tensor<1x1x2x2x2xf32>
    %3 = stablehlo.multiply %arg6, %arg6 : tensor<1x1x2x2xf32>
    return %0, %1, %2, %3 : tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>, tensor<1x1x2x2xf32>
  }
  func.func @main(%arg0: tensor<2x2x2x1x1xf32>, %arg1: tensor<2x2x2x1x1xf32>, %arg2: tensor<2x2x2x1x1xf32>, %arg3: tensor<2x2x1x1xf32>) -> (tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>, tensor<2x2x1x1xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [4, 3, 2, 1, 0] : (tensor<2x2x2x1x1xf32>) -> tensor<1x1x2x2x2xf32>
    %1 = stablehlo.transpose %arg1, dims = [4, 3, 2, 1, 0] : (tensor<2x2x2x1x1xf32>) -> tensor<1x1x2x2x2xf32>
    %2 = stablehlo.transpose %arg2, dims = [4, 3, 2, 1, 0] : (tensor<2x2x2x1x1xf32>) -> tensor<1x1x2x2x2xf32>
    %3 = stablehlo.transpose %arg3, dims = [3, 2, 1, 0] : (tensor<2x2x1x1xf32>) -> tensor<1x1x2x2xf32>
    %c = stablehlo.constant dense<false> : tensor<1x1x2x2xi1>
    %4:2 = stablehlo.custom_call @mps.metal_kernel_lib(%0, %1, %2, %3, %c) {backend_config = "{\22name\22:\22_fwd\22,\22metallib_path\22:\22/tmp/none.metallib\22,\22grid\22:[1,1,1],\22threadgroup\22:[1024,1,1],\22dispatch\22:\22threadgroups\22,\22buffers\22:[{\22slot\22:0,\22kind\22:\22output\22,\22arg\22:0},{\22slot\22:1,\22kind\22:\22output\22,\22arg\22:1},{\22slot\22:2,\22kind\22:\22input\22,\22arg\22:0},{\22slot\22:3,\22kind\22:\22input\22,\22arg\22:1},{\22slot\22:4,\22kind\22:\22input\22,\22arg\22:2},{\22slot\22:5,\22kind\22:\22input\22,\22arg\22:3},{\22slot\22:6,\22kind\22:\22input\22,\22arg\22:4}]}", enzyme.active_operands = array<i64: 0, 1, 2, 3>, enzyme.reverse = @"##_fwd_reverse#281"} : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xi1>) -> (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2xf32>)
    %5 = stablehlo.transpose %4#0, dims = [4, 3, 2, 1, 0] : (tensor<1x1x2x2x2xf32>) -> tensor<2x2x2x1x1xf32>
    %6 = stablehlo.transpose %0, dims = [4, 3, 2, 1, 0] : (tensor<1x1x2x2x2xf32>) -> tensor<2x2x2x1x1xf32>
    %7 = stablehlo.transpose %1, dims = [4, 3, 2, 1, 0] : (tensor<1x1x2x2x2xf32>) -> tensor<2x2x2x1x1xf32>
    %8 = stablehlo.transpose %2, dims = [4, 3, 2, 1, 0] : (tensor<1x1x2x2x2xf32>) -> tensor<2x2x2x1x1xf32>
    %9 = stablehlo.transpose %3, dims = [3, 2, 1, 0] : (tensor<1x1x2x2xf32>) -> tensor<2x2x1x1xf32>
    return %5, %6, %7, %8, %9 : tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>, tensor<2x2x1x1xf32>
  }
}