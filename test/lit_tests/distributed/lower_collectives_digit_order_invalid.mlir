// RUN: enzymexlamlir-opt --distributed-lower-collectives -verify-diagnostics %s

// A peel of the fast extent-4 atom p0 around a reduce-scatter on p1, over a
// 2x2x2 tile (already atomic): every local digit has extent 2, so no digit
// can hold the peel's shard, and the collective is reported as one that
// cannot be lowered.
module @negative_peel_without_whole_digit {
  %0 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %1 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %2 = axis.product (%0 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %3 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %4 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %5 = axis.product (%3 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %4 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %6 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %7 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %8 = axis.product (%6 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %9 = axis.factor %29 : !axis.shape_axis<tensor<2x2x2xi8>, 0><2, 1>
  %10 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %11 = axis.product (%9 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 0>, 2, 1>)
  %12 = axis.product (%10 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %13 = axis.factor %30 : !axis.shape_axis<tensor<2x2x2xi8>, 1><2, 1>
  %14 = axis.factor %32 : !axis.shape_axis<tensor<2x2xi8>, 0><2, 1>
  %15 = axis.product (%13 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 1>, 2, 1>)
  %16 = axis.product (%14 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 0>, 2, 1>)
  %17 = axis.factor %31 : !axis.shape_axis<tensor<2x2x2xi8>, 2><2, 1>
  %18 = axis.factor %33 : !axis.shape_axis<tensor<2x2xi8>, 1><2, 1>
  %19 = axis.product (%17 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 2>, 2, 1>)
  %20 = axis.product (%18 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 1>, 2, 1>)
  %21 = distributed.ReplicationAxis 4 : <4>
  %22 = axis.factor %21 : !distributed.replication_axis<4><4, 1>
  %23 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %24 = axis.product (%22 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %25 = axis.product (%23 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %26 = axis.map %11, %15, %19, %24 to %12, %16, %20, %25 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>]
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [8.000000e+00, 1.000000e-01]}
  func.func @main() {
    return
  }
  %27:2 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<2, 1>
  %28 = distributed.ReplicationAxis 4 : <4>
  %29 = axis.getaxis tensor<2x2x2xi8> 0
  %30 = axis.getaxis tensor<2x2x2xi8> 1
  %31 = axis.getaxis tensor<2x2x2xi8> 2
  %32 = axis.getaxis tensor<2x2xi8> 0
  %33 = axis.getaxis tensor<2x2xi8> 1
  %34 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %35 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %36 = axis.product (%34 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %35 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %37 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %38 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %39 = axis.product (%37 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %38 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %40 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %41 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %42 = axis.product (%40 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>, %41 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %43 = axis.factor %29 : !axis.shape_axis<tensor<2x2x2xi8>, 0><2, 1>
  %44 = axis.product (%43 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 0>, 2, 1>)
  %45 = axis.factor %27#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %46 = axis.product (%45 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %47 = axis.factor %30 : !axis.shape_axis<tensor<2x2x2xi8>, 1><2, 1>
  %48 = axis.product (%47 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 1>, 2, 1>)
  %49 = axis.factor %32 : !axis.shape_axis<tensor<2x2xi8>, 0><2, 1>
  %50 = axis.product (%49 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 0>, 2, 1>)
  %51 = axis.factor %31 : !axis.shape_axis<tensor<2x2x2xi8>, 2><2, 1>
  %52 = axis.product (%51 : !axis.axis_factor<!axis.shape_axis<tensor<2x2x2xi8>, 2>, 2, 1>)
  %53 = axis.factor %33 : !axis.shape_axis<tensor<2x2xi8>, 1><2, 1>
  %54 = axis.product (%53 : !axis.axis_factor<!axis.shape_axis<tensor<2x2xi8>, 1>, 2, 1>)
  %55 = axis.factor %28 : !distributed.replication_axis<4><4, 1>
  %56 = axis.product (%55 : !axis.axis_factor<!distributed.replication_axis<4>, 4, 1>)
  %57 = axis.factor %27#0 : !distributed.physical_comm_axis<4, 2><4, 1>
  %58 = axis.product (%57 : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %59 = axis.map %44, %48, %52, %56 to %46, %50, %54, %58 : [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<2>, !axis.factor_group<4>]
  %60 = tensor.empty() : tensor<2x2x2xi8>
  // expected-error @+1 {{cannot lower the peeled all-reduce over mesh axis 0 atom 0: no local digit that stays local until the peel closes is divisible by its extent 4}}
  %61 = distributed.Collective %60 : tensor<2x2x2xi8> on %2 : <8> to tensor<2x2xi8> on %5 : <8> reduces (%8 : !axis.factor_group<8>) maps %26 : !axis.map {
  ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
    %63 = stablehlo.add %arg0, %arg1 : tensor<i8>
    stablehlo.return %63 : tensor<i8>
  }
  %62 = distributed.Await %61 : <tensor<2x2xi8>> -> tensor<2x2xi8>
}

