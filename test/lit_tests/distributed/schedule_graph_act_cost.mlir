// RUN: enzymexlamlir-opt --split-input-file --distributed-print-schedule-graph %s 2>&1 >/dev/null | FileCheck %s

// A func.call whose resolved callee carries act.cost (ACT's convention for a
// bodyless accelerator-launch declaration, see the ACT plan doc's real
// sample) costs as one atomic Compute node instead of recursing into a
// callee body: there is none to recurse into.
//
// This case keeps the DistributedKernel wrapper (no distributed-inline-kernels
// in the RUN line): a kernel body whose region has been spliced with ACT's
// patched form is exactly what search-time candidate scoring costs, since
// ACT splicing must happen for every candidate and runs before the
// finalizeSchedule-gated distributed-inline-kernels (D4) step ever
// dissolves the wrapper (see the ACT plan doc's "resolved design fork").
// buildKernelNodes (the still-wrapped code path) therefore needs the same
// act.cost special case as buildBlockNodes (the dissolved-block-scope code
// path, exercised by the next case below); memref staging around the launch
// is elided from this minimal fixture since ScheduleGraph only cares about
// the call's own SSA operands, not the staging ops around it.
//
// device_clock_frequency = 100 converts act.cost's 480 cycles into an
// isolated duration of 4.8; computeCost.flops is back-solved to
// device.flops (2000, the default) * 4.8 = 9600 so the call still shows up
// as full compute-resource demand (see appendActLaunchNode's comment).
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: compute func.call: flops=9600 bytes=0 time=4.8 preds=[0]
// CHECK: schedule graph: 2 nodes (2 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {device_clock_frequency = 1.0e2}

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %k = distributed.DistributedKernel (%a : tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<4xf32>):
      %s = stablehlo.add %x, %x : tensor<4xf32>
      %r = func.call @act_launch(%s) : (tensor<4xf32>) -> tensor<4xf32>
      distributed.DistributedYield (%r : tensor<4xf32>)
    }
    return %k : tensor<4xf32>
  }

  func.func private @act_launch(tensor<4xf32>) -> tensor<4xf32> attributes {act.cost = 4.8e2 : f64}
}

// -----

// Same op sequence as above, but with no DistributedKernel wrapper: the
// shape distributed-inline-kernels (D4) would produce by dissolving one.
// This exercises buildBlockNodes's own act.cost special case (a separate
// code path from buildKernelNodes above, since buildBlockNodes's dispatch
// loop is never reached while a kernel wrapper is still in place) and
// confirms it costs identically to the still-wrapped case.
// CHECK: node 0: compute stablehlo.add: flops=4 bytes=48 time=2.4 preds=[]
// CHECK: node 1: compute func.call: flops=9600 bytes=0 time=4.8 preds=[0]
// CHECK: schedule graph: 2 nodes (2 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {device_clock_frequency = 1.0e2}

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %s = stablehlo.add %a, %a : tensor<4xf32>
    %r = func.call @act_launch(%s) : (tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }

  func.func private @act_launch(tensor<4xf32>) -> tensor<4xf32> attributes {act.cost = 4.8e2 : f64}
}

// -----

// No device_clock_frequency given: DeviceCostParams::kDefaultClockFrequency
// (1) treats act.cost's cycle count as already being in this model's time
// unit, so isolatedDuration is exactly the 100-cycle act.cost, unconverted.
// CHECK: node 0: compute func.call: flops=200000 bytes=0 time=100 preds=[]
// CHECK: schedule graph: 1 nodes (1 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %r = func.call @act_launch(%a) : (tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }

  func.func private @act_launch(tensor<4xf32>) -> tensor<4xf32> attributes {act.cost = 1.0e2 : f64}
}

// -----

// A call to a bodyless callee that carries no act.cost (ACT's convention for
// an @act_stage_N/@act_fetch_N declaration, tagged act.role instead) falls
// through to the hard-barrier path (buildCallNode), which requires a
// walkable single-block callee region; a bodyless declaration has none, so
// the whole graph is infeasible. A real spliced kernel never calls a
// stage/fetch declaration this way: act_launch's own operands are the staged
// copies (see the ACT plan doc's "Redone breakdown" item 1).
// CHECK: schedule graph: infeasible (call to an unresolved or non-single-block callee)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %r = func.call @act_stage_0(%a) : (tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }

  func.func private @act_stage_0(tensor<4xf32>) -> tensor<4xf32> attributes {act.role = "stage"}
}
