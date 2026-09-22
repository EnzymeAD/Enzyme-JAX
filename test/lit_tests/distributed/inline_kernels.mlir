// RUN: enzymexlamlir-opt --split-input-file --distributed-inline-kernels %s | FileCheck %s

// A single kernel with no external dependents beyond its own return: the
// wrapper disappears and the body's ops splice into @main in its place,
// with the block argument replaced by the kernel's own operand and the
// returned value replaced by the yielded one.
// CHECK-LABEL: func.func @single_kernel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[D:.*]] = stablehlo.add %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT: return %[[D]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NOT: distributed.DistributedKernel
func.func @single_kernel(%a: tensor<4xf32>) -> tensor<4xf32> {
  %k = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %d = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
  return %k : tensor<4xf32>
}

// -----

// A kernel whose result feeds a distributed.Collective: once inlined, the
// collective's input_object operand must point directly at the inlined
// body's yielded value.
//
// Axis-algebra metadata ops sit directly in the module body rather than
// inside a function, matching the atomic-collective examples elsewhere in
// this directory (e.g. schedule_graph.mlir).
// CHECK-LABEL: module {
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EMPTY]], %[[EMPTY]] : tensor<4xf32>
// CHECK-NEXT: distributed.Collective %[[ADD]] : tensor<4xf32> on
// CHECK-NOT: distributed.DistributedKernel
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }

  %p0 = distributed.GetPhysicalMeshAxes @mesh0 : !distributed.physical_comm_axis<2, 1>
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

  %a = tensor.empty() : tensor<4xf32>
  %k1 = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %s = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%s : tensor<4xf32>)
  }

  %h = distributed.Collective %k1 : tensor<4xf32> on %mesh_in : !axis.factor_group<2> to tensor<4xf32> on %mesh_out : !axis.factor_group<2> reduces (%red0 : !axis.factor_group<2>) maps %map : !axis.map {
  ^bb0(%lhs_v: tensor<f32>, %rhs_v: tensor<f32>):
    %r = stablehlo.add %lhs_v, %rhs_v : tensor<f32>
    stablehlo.return %r : tensor<f32>
  }
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<4xf32>> -> tensor<4xf32>
}

// -----

// Multiple kernels in sequence: after inlining, the second kernel's body
// must reference the first kernel's yielded value, since the first
// kernel's own result no longer exists once it is erased.
// CHECK-LABEL: func.func @sequential_kernels(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[D:.*]] = stablehlo.add %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT: %[[E:.*]] = stablehlo.add %[[D]], %[[D]] : tensor<4xf32>
// CHECK-NEXT: return %[[E]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NOT: distributed.DistributedKernel
func.func @sequential_kernels(%a: tensor<4xf32>) -> tensor<4xf32> {
  %k1 = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%x: tensor<4xf32>):
    %d = stablehlo.add %x, %x : tensor<4xf32>
    distributed.DistributedYield (%d : tensor<4xf32>)
  }
  %k2 = distributed.DistributedKernel (%k1 : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
    axes () {
  ^bb0(%y: tensor<4xf32>):
    %e = stablehlo.add %y, %y : tensor<4xf32>
    distributed.DistributedYield (%e : tensor<4xf32>)
  }
  return %k2 : tensor<4xf32>
}
