module {
  sdy.mesh @mesh = <["x"=2]>
  distributed.AxisAllToAll @ax1 2 1600000000 0
  distributed.PhysicalMesh @phys_mesh [@ax1]
  func.func @innerouterinneroperatorparallel(%arg0: tensor<512x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %arg1: tensor<1024x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<1x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg3: tensor<8192x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<512x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
    %0 = distributed.AxisFactor @ax1 [2 : i32]
    %1 = distributed.LogicalMesh %0 {enzyme.shardy_axis_names = ["x"], physical_mesh = @phys_mesh}
    %2 = distributed.MeshComputation %1 %arg0, %arg1, %arg2, %arg3 tensor<512x1024xf32>, tensor<1024x64xf32>, tensor<1x128xf32>, tensor<8192x10xf32> -> tensor<512x10xf32> {
    ^bb0(%arg4: tensor<512x1024xf32>, %arg5: tensor<1024x64xf32>, %arg6: tensor<1x128xf32>, %arg7: tensor<8192x10xf32>):
      %3 = distributed.LogicalMesh %0 {physical_mesh = @phys_mesh}
      %4 = "distributed.Sharding"() : () -> !distributed.sharding
      %5 = "distributed.Sharding"(%0) : (!distributed.logical_comm_axis) -> !distributed.sharding
      %6 = distributed.Collective %1(tensor<512x64xf32>> tensor<512x64xf32>, %4, %4) -> (tensor<512x64xf32>> tensor<512x64xf32>, %5, %4)
      %8 = distributed.Collective %1(tensor<8192x10xf32>> tensor<8192x10xf32>, %5, %4) -> (tensor<8192x10xf32>> tensor<8192x10xf32>, %4, %4)
      %10 = distributed.RegionComputation %3 %arg4, %arg5, %arg6, %arg7 tensor<512x1024xf32>, tensor<1024x64xf32>, tensor<1x128xf32>, tensor<8192x10xf32> -> tensor<512x10xf32> {
      ^bb0(%arg8: tensor<512x1024xf32>, %arg9: tensor<1024x64xf32>, %arg10: tensor<1x128xf32>, %arg11: tensor<8192x10xf32>):
        %11 = stablehlo.dot %arg8, %arg9 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<512x1024xf32>, tensor<1024x64xf32>) -> tensor<512x64xf32>
        %12 = sdy.all_reduce {"x"} %11 out_sharding=<@mesh, [{}, {}]> : tensor<512x64xf32>
        %13 = distributed.SendRecv %6 %12 tensor<512x64xf32> -> tensor<512x64xf32> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
        %14 = stablehlo.reshape %13 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<512x64xf32>) -> tensor<32768x1xf32>
        %15 = stablehlo.dot %14, %arg10 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<32768x1xf32>, tensor<1x128xf32>) -> tensor<32768x128xf32>
        %16 = stablehlo.reshape %15 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<32768x128xf32>) -> tensor<512x8192xf32>
        // some dummy ops to make the timing more interesting
        %116 = stablehlo.cosine %16 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<512x8192xf32>) -> tensor<512x8192xf32>
        %17 = distributed.SendRecv %8 %arg11 tensor<8192x10xf32> -> tensor<8192x10xf32> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
        %18 = stablehlo.dot %116, %17 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<512x8192xf32>, tensor<8192x10xf32>) -> tensor<512x10xf32>
        distributed.DistributedYield %18 tensor<512x10xf32>
      }, {
      ^bb0(%arg8: tensor<512x1024xf32>, %arg9: tensor<1024x64xf32>, %arg10: tensor<1x128xf32>, %arg11: tensor<8192x10xf32>):
        %11 = stablehlo.dot %arg8, %arg9 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<512x1024xf32>, tensor<1024x64xf32>) -> tensor<512x64xf32>
        // some dummy ops to make the timing more interesting
        %111 = stablehlo.add %11, %11 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<512x64xf32>, tensor<512x64xf32>) -> tensor<512x64xf32>
        %211 = stablehlo.abs %111 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<512x64xf32>) -> tensor<512x64xf32>
        %12 = sdy.all_reduce {"x"} %211 out_sharding=<@mesh, [{}, {}]> : tensor<512x64xf32>
        %13 = distributed.SendRecv %6 %12 tensor<512x64xf32> -> tensor<512x64xf32> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
        %14 = stablehlo.reshape %13 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<512x64xf32>) -> tensor<32768x1xf32>
        %15 = stablehlo.dot %14, %arg10 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<32768x1xf32>, tensor<1x128xf32>) -> tensor<32768x128xf32>
        %16 = stablehlo.reshape %15 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<32768x128xf32>) -> tensor<512x8192xf32>
        %17 = distributed.SendRecv %8 %arg11 tensor<8192x10xf32> -> tensor<8192x10xf32> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
        %18 = stablehlo.dot %16, %17 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<512x8192xf32>, tensor<8192x10xf32>) -> tensor<512x10xf32>
        distributed.DistributedYield %18 tensor<512x10xf32>
      }
      distributed.DistributedYield %10 tensor<512x10xf32>
    }
    return %2 : tensor<512x10xf32>
  }
}