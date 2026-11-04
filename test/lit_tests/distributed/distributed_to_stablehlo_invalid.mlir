// RUN: enzymexlamlir-opt --split-input-file --distributed-to-stablehlo --verify-diagnostics %s

// distributed-to-stablehlo may run on IR that isn't a finalized schedule, so
// whatever it can't translate is a diagnostic. Each case is otherwise a
// finalized-form module on a 2x2 mesh.

// A declared sharding over a replication axis has no HloSharding.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %replication = distributed.ReplicationAxis 2 : <2>
  %replication_factor = axis.factor %replication : !distributed.replication_axis<2><2, 1>
  %over_replication = axis.product (%replication_factor : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  // expected-error @+1 {{expected partitioning axes over physical mesh and device-local axes, found '!distributed.replication_axis<2>'}}
  "distributed.DistributedFunction"(%over_replication) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A dimension split over a device-local factor and then a physical one gives
// each device strided elements, which a tiled HloSharding can't express.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %local = distributed.DeviceLocalAxis 2 : <2>
  %local_factor = axis.factor %local : !distributed.device_local_axis<2><2, 1>
  %over_local = axis.product (%local_factor : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  // expected-error @+1 {{expected the physical factors of each dimension to be major to its device-local factors}}
  "distributed.DistributedFunction"(%over_local, %over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0, 1]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// An argument cast must give the local type its declared sharding does: here
// the cast also splits over a device-local factor the declaration keeps whole.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %local = distributed.DeviceLocalAxis 2 : <2>
  %local_factor = axis.factor %local : !distributed.device_local_axis<2><2, 1>
  %over_local = axis.product (%local_factor : !axis.axis_factor<!distributed.device_local_axis<2>, 2, 1>)
  %unsplit = axis.product ()
  "distributed.DistributedFunction"(%unsplit) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<2xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    // expected-error @+1 {{argument #0 has a cast to 'tensor<2xf32>' {replicated}, but its declared layout is 'tensor<4xf32>' {replicated}}}
    %0 = distributed.CastGlobalToLocal %arg0 axes (%over_local : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    distributed.DistributedYield (%0 : tensor<2xf32>)
  }) : (!axis.factor_group<1>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// An argument cast must give the HloSharding its declared sharding does, not
// only the local type: here over the minor axis instead of the major one.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %minor = axis.factor %axes#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %over_minor = axis.product (%minor : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  "distributed.DistributedFunction"(%over_major, %over_minor) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<2xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    // expected-error @+1 {{argument #0 has a cast to 'tensor<2xf32>' {devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}, but its declared layout is 'tensor<2xf32>' {devices=[2,2]<=[4] last_tile_dim_replicate}}}
    %0 = distributed.CastGlobalToLocal %arg0 axes (%over_minor : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    distributed.DistributedYield (%0 : tensor<2xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A result cast must match the result's declared sharding.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %minor = axis.factor %axes#1 : !distributed.physical_comm_axis<2, 1><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  %over_minor = axis.product (%minor : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  "distributed.DistributedFunction"(%over_major, %over_minor) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<2xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<2xf32>):
    // expected-error @+1 {{result #0 has a cast from 'tensor<2xf32>' {devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}, but its declared layout is 'tensor<2xf32>' {devices=[2,2]<=[4] last_tile_dim_replicate}}}
    %0 = distributed.CastLocalToGlobal %arg0 axes (%over_minor : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    distributed.DistributedYield (%0 : tensor<4xf32>)
  }) : (!axis.factor_group<2>, !axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A sharded argument has no global value in the per-device program.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  // expected-error @+1 {{argument #0 is sharded but used at its global type}}
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> (tensor<2xf32>, tensor<4xf32>), output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %0 = distributed.CastGlobalToLocal %arg0 axes (%over_major : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    %1 = stablehlo.negate %arg0 : tensor<4xf32>
    distributed.DistributedYield (%0 : tensor<2xf32>, %1 : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// Returning an argument directly keeps its layout, so the result must declare
// the same one.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  // expected-error @+1 {{result #0 returns argument #0, whose declared layout differs}}
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A sharded result needs a cast from its local value.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  // expected-error @+1 {{result #0 is sharded but has no CastLocalToGlobal}}
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %0 = stablehlo.negate %arg0 : tensor<4xf32>
    distributed.DistributedYield (%0 : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A partial sum at an entry result, unreduced over a digit that splits
// devices, has no HloSharding.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  // expected-error @+1 {{result #0 is unreduced; exporting needs reduced entry results}}
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = [0]>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// An unlowered collective: the schedule wasn't finalized.
module {
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  %p0 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %in = tensor.empty() : tensor<4xf32>
  // expected-error @+1 {{unlowered distributed.Collective; export a schedule finalized by distributed-search-lowering-pipeline}}
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// An unlowered Await is reported too (here placed before its collective in
// the module's graph region, so the walk reaches it first).
module {
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  // expected-error @+1 {{unlowered distributed.Await; export a schedule finalized by distributed-search-lowering-pipeline}}
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
  %p0 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 1>
  %rep2 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %ti0 = axis.getaxis tensor<4xf32> 0
  %to0 = axis.getaxis tensor<4xf32> 0
  %f1 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_in = axis.product (%f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f2 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %mesh_out = axis.product (%f2 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f3 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %red0 = axis.product (%f3 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %f4 = axis.factor %ti0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %lhs0 = axis.product (%f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f5 = axis.factor %to0 : !axis.shape_axis<tensor<4xf32>, 0> <4, 1>
  %rhs0 = axis.product (%f5 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>)
  %f6 = axis.factor %rep2 : !distributed.replication_axis<2> <2, 1>
  %lhs1 = axis.product (%f6 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %f7 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %rhs1 = axis.product (%f7 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs0, %lhs1 to %rhs0, %rhs1 : [!axis.factor_group<4>, !axis.factor_group<2>] [!axis.factor_group<4>, !axis.factor_group<2>]
  %in = tensor.empty() : tensor<4xf32>
  %h = distributed.Collective %in : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
}

// -----

// Any other distributed op, here an anchor, has no StableHLO form.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<2xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %0 = distributed.CastGlobalToLocal %arg0 axes (%over_major : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    // expected-error @+1 {{unexpected distributed.AnchorPartitioning in a schedule to export}}
    %1 = distributed.AnchorPartitioning %0 axes (%over_major : !axis.factor_group<2>) : tensor<2xf32>
    distributed.DistributedYield (%1 : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A leftover Shardy op, here a sharding constraint, has no per-device form.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %unsplit = axis.product ()
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  "distributed.DistributedFunction"(%unsplit) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    // expected-error @+1 {{unexpected sdy.sharding_constraint in a schedule to export}}
    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : tensor<4xf32>
    distributed.DistributedYield (%0 : tensor<4xf32>)
  }) : (!axis.factor_group<1>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// Device ids 0, 2 on a single axis of id stride 2 aren't 0..N-1.
module {
  // expected-error @+1 {{expected the physical axes to number devices densely as 0..N-1}}
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>]
}

// -----

// The export needs the entry function @main.
// expected-error @+1 {{expected a distributed.DistributedFunction named @main}}
module {
  "distributed.DistributedFunction"() <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<4xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "f"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    distributed.DistributedYield (%arg0 : tensor<4xf32>)
  }) : () -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A cast of a value other than an argument is a layout change inside the
// body, which the export can't express.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<4xf32>) -> tensor<2xf32>, output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %0 = stablehlo.negate %arg0 : tensor<4xf32>
    // expected-error @+1 {{expected a cast only at a function boundary: on an argument, or feeding only the function's results}}
    %1 = distributed.CastGlobalToLocal %0 axes (%over_major : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    distributed.DistributedYield (%1 : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// A result cast whose global value is also used in the body isn't only at the
// boundary.
module {
  %axes:2 = distributed.GetPhysicalMeshAxes @m : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %major = axis.factor %axes#0 : !distributed.physical_comm_axis<2, 2><2, 1>
  %over_major = axis.product (%major : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>)
  "distributed.DistributedFunction"(%over_major) <{argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>, function_type = (tensor<2xf32>) -> (tensor<4xf32>, tensor<4xf32>), output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<2xf32>):
    // expected-error @+1 {{expected a cast only at a function boundary: on an argument, or feeding only the function's results}}
    %0 = distributed.CastLocalToGlobal %arg0 axes (%over_major : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    %1 = stablehlo.negate %0 : tensor<4xf32>
    distributed.DistributedYield (%0 : tensor<4xf32>, %1 : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
}

// -----

// The export needs the physical mesh for the device count.
// expected-error @+1 {{expected exactly one distributed physical mesh in module, found 0}}
module {
}

// -----

// Several physical meshes leave the device count ambiguous.
// expected-error @+1 {{expected exactly one distributed physical mesh in module, found 2}}
module {
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  distributed.PhysicalMesh @n device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
}
