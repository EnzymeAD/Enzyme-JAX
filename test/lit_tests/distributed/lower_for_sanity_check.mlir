// RUN: enzymexlamlir-opt --split-input-file --distributed-lower-for-sanity-check %s | FileCheck %s

// Single physical axis, one trivial kernel, no collectives: expect one
// stablehlo.while, a reshape materializing the expanded input outside it,
// dynamic_slice/dynamic_update_slice inside, and a plain
// func.func (no distributed.*/axis.* residue) at the top level.
// CHECK-LABEL: func.func @main
// CHECK-NOT: distributed.
// CHECK-NOT: axis.
// CHECK: stablehlo.reshape
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.add
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %axes_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<2>) : tensor<4xf32> -> tensor<2xf32>
    %kres = distributed.DistributedKernel (%local_in : tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<2xf32>):
      %c = stablehlo.add %a, %a : tensor<2xf32>
      distributed.DistributedYield (%c : tensor<2xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %kres axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<4xf32>
    distributed.DistributedYield (%global_out : tensor<4xf32>)
  }) : (!axis.factor_group<2>) -> ()
}

// -----

// Two physical axes sharding one kernel: expect two properly *nested*
// stablehlo.while ops (not a flat loop with decode math), each loop's own
// induction variable used directly as that axis's slice coordinate.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.dynamic_update_slice
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]

  %p0, %p1 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 2> <2, 1>
  %f1 = axis.factor %p1 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %axes_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 2>, 2, 1>, %f1 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
    %kres = distributed.DistributedKernel (%local_in : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<1xf32>):
      %c = stablehlo.add %a, %a : tensor<1xf32>
      distributed.DistributedYield (%c : tensor<1xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %kres axes (%axes_grp : !axis.factor_group<4>) : tensor<1xf32> -> tensor<4xf32>
    distributed.DistributedYield (%global_out : tensor<4xf32>)
  }) : (!axis.factor_group<4>) -> ()
}

// -----

// An all-reduce-shaped collective between two kernels: reduces() is
// non-empty, so expect a stablehlo.reduce (classified from the cloned
// add-body via its identity element -- see
// stablehlo::classifyReduceBlockKind/getIdentityValueForReduceKind) over
// the reduced mesh axis, and no hardware-collective machinery at all (this
// pass never reuses DistributedToHlo.cpp's patterns).
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.reduce
// CHECK-SAME: applies stablehlo.add
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.while
// CHECK-NOT: stablehlo.all_reduce
// CHECK-NOT: stablehlo.async_start
// CHECK-NOT: channel_handle
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %axes_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %r0 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rf0 = axis.factor %r0 : !distributed.replication_axis<2> <2, 1>
  %lhs = axis.product (%rf0 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<2xf32>) -> tensor<2xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<2xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
    %k1 = distributed.DistributedKernel (%local_in : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<1xf32>):
      %c = stablehlo.multiply %a, %a : tensor<1xf32>
      distributed.DistributedYield (%c : tensor<1xf32>)
    }
    %g1 = distributed.CastLocalToGlobal %k1 axes (%axes_grp : !axis.factor_group<2>) : tensor<1xf32> -> tensor<2xf32>
    %l1 = distributed.CastGlobalToLocal %g1 axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
    %h = distributed.Collective %l1 : tensor<1xf32> on %axes_grp : !axis.factor_group<2> to tensor<1xf32> on %axes_grp : !axis.factor_group<2> reduces (%axes_grp : !axis.factor_group<2>) maps %map : !axis.map {
    ^bb2(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
      %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
    %k2 = distributed.DistributedKernel (%v : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb3(%b: tensor<1xf32>):
      %d = stablehlo.add %b, %b : tensor<1xf32>
      distributed.DistributedYield (%d : tensor<1xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %k2 axes (%axes_grp : !axis.factor_group<2>) : tensor<1xf32> -> tensor<2xf32>
    distributed.DistributedYield (%global_out : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
}

// -----

// An all-gather/broadcast-shaped collective with an empty reduces() list:
// expect a broadcast_in_dim materializing the collective's output directly
// (no reduction fold in between the two kernels' own while loops).
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.while
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %real_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  // A trivial (zero-factor) group: extent 1, "not divided along anything"
  // -- used for the kernel operand/result that stays fully replicated
  // (identical on every device) up to the collective, with no real
  // per-tensor-dimension split at all.
  %trivial_grp = axis.product ()

  %r0 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rf0 = axis.factor %r0 : !distributed.replication_axis<2> <2, 1>
  %lhs = axis.product (%rf0 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  "distributed.DistributedFunction"(%real_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    function_type = (tensor<1xf32>) -> (tensor<2xf32>),
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<1xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%trivial_grp : !axis.factor_group<1>) : tensor<1xf32> -> tensor<1xf32>
    %k1 = distributed.DistributedKernel (%local_in : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<1xf32>):
      %c = stablehlo.multiply %a, %a : tensor<1xf32>
      distributed.DistributedYield (%c : tensor<1xf32>)
    }
    %g1 = distributed.CastLocalToGlobal %k1 axes (%trivial_grp : !axis.factor_group<1>) : tensor<1xf32> -> tensor<1xf32>
    %l1 = distributed.CastGlobalToLocal %g1 axes (%trivial_grp : !axis.factor_group<1>) : tensor<1xf32> -> tensor<1xf32>
    %h = distributed.Collective %l1 : tensor<1xf32> on %trivial_grp : !axis.factor_group<1> to tensor<1xf32> on %real_grp : !axis.factor_group<2> reduces () maps %map : !axis.map
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
    %k2 = distributed.DistributedKernel (%v : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb3(%b: tensor<1xf32>):
      %d = stablehlo.add %b, %b : tensor<1xf32>
      distributed.DistributedYield (%d : tensor<1xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %k2 axes (%real_grp : !axis.factor_group<2>) : tensor<1xf32> -> tensor<2xf32>
    distributed.DistributedYield (%global_out : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
}

// -----

// A gather-shaped collective: the mapping's rhs is a ShapeAxisType factor
// of the output tensor's own dim 0 (Physical(axis0) -> TensorDim(0)) rather
// than another physical axis, so the physical axis's genuinely-varying
// data ends up relabeled onto that tensor dimension instead of staying a
// mesh axis. The input tile's extent-1 dim is dropped in the process. Since
// the result is no longer split by any physical axis, it is broadcast
// uniformly across the mesh before the second kernel's loop.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.while
// CHECK-NOT: stablehlo.all_gather
// CHECK-NOT: channel_handle
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %real_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %trivial_grp = axis.product ()

  %ta = axis.getaxis tensor<2xf32> 0
  %tf0 = axis.factor %ta : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>

  %lhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %rhs = axis.product (%tf0 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  "distributed.DistributedFunction"(%real_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<2xf32>) -> (tensor<2xf32>),
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<2xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%real_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
    %k1 = distributed.DistributedKernel (%local_in : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<1xf32>):
      %c = stablehlo.multiply %a, %a : tensor<1xf32>
      distributed.DistributedYield (%c : tensor<1xf32>)
    }
    %g1 = distributed.CastLocalToGlobal %k1 axes (%real_grp : !axis.factor_group<2>) : tensor<1xf32> -> tensor<2xf32>
    %l1 = distributed.CastGlobalToLocal %g1 axes (%real_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
    %h = distributed.Collective %l1 : tensor<1xf32> on %real_grp : !axis.factor_group<2> to tensor<2xf32> on %trivial_grp : !axis.factor_group<1> reduces () maps %map : !axis.map
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<2xf32>> -> tensor<2xf32>
    %k2 = distributed.DistributedKernel (%v : tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb3(%b: tensor<2xf32>):
      %d = stablehlo.add %b, %b : tensor<2xf32>
      distributed.DistributedYield (%d : tensor<2xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %k2 axes (%trivial_grp : !axis.factor_group<1>) : tensor<2xf32> -> tensor<2xf32>
    distributed.DistributedYield (%global_out : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
}

// -----

// A collective fed directly by a kernel result, with no cast in between
// (nothing on that edge changes scope or shape). The kernel result must still
// be expanded per mesh coordinate so the collective sees distinct values;
// expect the same reduce-then-map recipe as the cast-bounded case above.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.reduce
// CHECK-SAME: applies stablehlo.add
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.while
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
  %f0 = axis.factor %p0 : !distributed.physical_comm_axis<2, 1> <2, 1>
  %axes_grp = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)

  %r0 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rf0 = axis.factor %r0 : !distributed.replication_axis<2> <2, 1>
  %lhs = axis.product (%rf0 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %rhs = axis.product (%f0 : !axis.axis_factor<!distributed.physical_comm_axis<2, 1>, 2, 1>)
  %map = axis.map %lhs to %rhs : [!axis.factor_group<2>] [!axis.factor_group<2>]

  "distributed.DistributedFunction"(%axes_grp) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<2xf32>) -> tensor<2xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<2xf32>):
    %local_in = distributed.CastGlobalToLocal %arg0 axes (%axes_grp : !axis.factor_group<2>) : tensor<2xf32> -> tensor<1xf32>
    %k1 = distributed.DistributedKernel (%local_in : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb1(%a: tensor<1xf32>):
      %c = stablehlo.multiply %a, %a : tensor<1xf32>
      distributed.DistributedYield (%c : tensor<1xf32>)
    }
    %h = distributed.Collective %k1 : tensor<1xf32> on %axes_grp : !axis.factor_group<2> to tensor<1xf32> on %axes_grp : !axis.factor_group<2> reduces (%axes_grp : !axis.factor_group<2>) maps %map : !axis.map {
    ^bb2(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
      %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
    %k2 = distributed.DistributedKernel (%v : tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        -> (tensor<1xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
        axes () {
    ^bb3(%b: tensor<1xf32>):
      %d = stablehlo.add %b, %b : tensor<1xf32>
      distributed.DistributedYield (%d : tensor<1xf32>)
    }
    %global_out = distributed.CastLocalToGlobal %k2 axes (%axes_grp : !axis.factor_group<2>) : tensor<1xf32> -> tensor<2xf32>
    distributed.DistributedYield (%global_out : tensor<2xf32>)
  }) : (!axis.factor_group<2>) -> ()
}

// -----

// A reduction over one half (stride 2) of a 4-way physical axis, leaving the
// other half (stride 1) mapped to itself. Splitting the axis into two atoms
// makes the reduction a plain reduce over one dim: expect a reshape exposing
// both halves, one reduce, and a broadcast restoring the reduced half.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.reshape {{.*}} -> tensor<2x2x1xf32>
// CHECK: stablehlo.reduce
// CHECK-SAME: dimensions = [0]
// CHECK: stablehlo.broadcast_in_dim
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %hi = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>
  %lo = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 1>
  %full = axis.product (%hi : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>, %lo : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)
  %hi_grp = axis.product (%hi : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %lo_grp = axis.product (%lo : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)
  %r0 = distributed.ReplicationAxis 2 : !distributed.replication_axis<2>
  %rf0 = axis.factor %r0 : !distributed.replication_axis<2> <2, 1>
  %repl_grp = axis.product (%rf0 : !axis.axis_factor<!distributed.replication_axis<2>, 2, 1>)
  %map = axis.map %lo_grp, %repl_grp to %lo_grp, %hi_grp : [!axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>]

  "distributed.DistributedFunction"(%full) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %l0 = distributed.CastGlobalToLocal %arg0 axes (%full : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
    %h = distributed.Collective %l0 : tensor<1xf32> on %full : !axis.factor_group<4> to tensor<1xf32> on %full : !axis.factor_group<4> reduces (%hi_grp : !axis.factor_group<2>) maps %map : !axis.map {
    ^bb2(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
      %sum = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
    %out = distributed.CastLocalToGlobal %v axes (%full : !axis.factor_group<4>) : tensor<1xf32> -> tensor<4xf32>
    distributed.DistributedYield (%out : tensor<4xf32>)
  }) : (!axis.factor_group<4>) -> ()
}

// -----

// A partial all-gather: only the stride-1 half of a 4-way physical axis is
// gathered into the tensor dimension, which keeps its own extent-2 tile as
// the minor part. The gathered half becomes the major part of the output dim.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.reshape {{.*}} -> tensor<2x2x2xf32>
// CHECK: stablehlo.broadcast_in_dim {{.*}} dims = [0, 2, 3]
// CHECK: stablehlo.reshape {{.*}} -> tensor<4x4xf32>
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 1>]

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 1>
  %hi = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 2>
  %lo = axis.factor %p0 : !distributed.physical_comm_axis<4, 1> <2, 1>
  %full = axis.product (%hi : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>, %lo : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)
  %hi_only = axis.product (%hi : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>)
  %none = axis.product ()
  %lo_grp = axis.product (%lo : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)

  %ta_in = axis.getaxis tensor<2xf32> 0
  %tin = axis.factor %ta_in : !axis.shape_axis<tensor<2xf32>, 0> <2, 1>
  %tin_grp = axis.product (%tin : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>)
  %ta = axis.getaxis tensor<4xf32> 0
  %tmajor = axis.factor %ta : !axis.shape_axis<tensor<4xf32>, 0> <2, 2>
  %tminor = axis.factor %ta : !axis.shape_axis<tensor<4xf32>, 0> <2, 1>
  %tmajor_grp = axis.product (%tmajor : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 2, 2>)
  %tminor_grp = axis.product (%tminor : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 2, 1>)
  %map = axis.map %lo_grp, %tin_grp to %tmajor_grp, %tminor_grp : [!axis.factor_group<2>, !axis.factor_group<2>] [!axis.factor_group<2>, !axis.factor_group<2>]

  "distributed.DistributedFunction"(%full) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<8xf32>) -> tensor<8xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<8xf32>):
    %l0 = distributed.CastGlobalToLocal %arg0 axes (%full : !axis.factor_group<4>) : tensor<8xf32> -> tensor<2xf32>
    %h = distributed.Collective %l0 : tensor<2xf32> on %lo_grp : !axis.factor_group<2> to tensor<4xf32> on %none : !axis.factor_group<1> reduces () maps %map : !axis.map
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
    %out = distributed.CastLocalToGlobal %v axes (%hi_only : !axis.factor_group<2>) : tensor<4xf32> -> tensor<8xf32>
    distributed.DistributedYield (%out : tensor<8xf32>)
  }) : (!axis.factor_group<4>) -> ()
}

// -----

// Aligning a mapping pair can cut an axis nothing else touches: the whole
// extent-4 factor of axis A maps onto the two halves of axis B, so A has to be
// split into two extent-2 atoms to pair up with them, even though no factor
// over A is split. Expect every mesh axis exposed as two atoms.
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.reshape {{.*}} -> tensor<2x2x2x2x1xf32>
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<4, 1>]

  %pa, %pb = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<4, 2>, !distributed.physical_comm_axis<4, 1>
  %fa = axis.factor %pa : !distributed.physical_comm_axis<4, 2> <4, 1>
  %bhi = axis.factor %pb : !distributed.physical_comm_axis<4, 1> <2, 2>
  %blo = axis.factor %pb : !distributed.physical_comm_axis<4, 1> <2, 1>
  %in_mesh = axis.product (%fa : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %out_mesh = axis.product (%bhi : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 2>, %blo : !axis.axis_factor<!distributed.physical_comm_axis<4, 1>, 2, 1>)
  %full = axis.product (%fa : !axis.axis_factor<!distributed.physical_comm_axis<4, 2>, 4, 1>)
  %map = axis.map %in_mesh to %out_mesh : [!axis.factor_group<4>] [!axis.factor_group<4>]

  "distributed.DistributedFunction"(%full) <{
    argument_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    function_type = (tensor<4xf32>) -> tensor<4xf32>,
    output_shardings = #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[0]] : unreduced_axes = []>]>,
    sym_name = "main"
  }> ({
  ^bb0(%arg0: tensor<4xf32>):
    %l0 = distributed.CastGlobalToLocal %arg0 axes (%full : !axis.factor_group<4>) : tensor<4xf32> -> tensor<1xf32>
    %h = distributed.Collective %l0 : tensor<1xf32> on %in_mesh : !axis.factor_group<4> to tensor<1xf32> on %out_mesh : !axis.factor_group<4> reduces () maps %map : !axis.map
    %v = distributed.Await %h : !distributed.asynch_handle<tensor<1xf32>> -> tensor<1xf32>
    %out = distributed.CastLocalToGlobal %v axes (%out_mesh : !axis.factor_group<4>) : tensor<1xf32> -> tensor<4xf32>
    distributed.DistributedYield (%out : tensor<4xf32>)
  }) : (!axis.factor_group<4>) -> ()
}
