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
// as full compute-resource demand (see appendActCalleeNode's comment).
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

// A call to a bodyless callee that carries act.role instead of act.cost
// (ACT's convention for an @act_stage_N/@act_fetch_N declaration: the memref
// copy it performs is costed as part of its neighboring launch, not
// separately) costs as a zero-duration atomic Compute node, the same
// non-barrier treatment as an act.cost-carrying launch above, rather than
// falling through to the hard-barrier path: a bodyless declaration has no
// walkable region for that path to recurse into, which would otherwise make
// the whole graph infeasible. A real spliced kernel's stage/fetch calls sit
// as plain top-level ops next to their launch, not only as the launch's own
// operand producers, so this path is genuinely reachable, not just
// defensive.
// CHECK: node 0: compute func.call: flops=0 bytes=0 time=0 preds=[]
// CHECK: schedule graph: 1 nodes (1 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %r = func.call @act_stage_0(%a) : (tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }

  func.func private @act_stage_0(tensor<4xf32>) -> tensor<4xf32> attributes {act.role = "stage"}
}
