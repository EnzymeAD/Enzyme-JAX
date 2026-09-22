// RUN: enzymexlamlir-opt --split-input-file --distributed-reorder-schedule %s | FileCheck %s

// Two independent kernels interleaved with a collective between them.
//
// Op order in the input: %a -> collective on %a -> %b (uses the collective's
// result) -> %c (independent of everything, textually last).
//
// %c has no dependency on anything, so once %a releases the single-issue
// compute engine, %c's admission overlaps the collective's transfer instead
// of waiting for %b's whole predecessor chain to finish: the simulator
// starts %c and the collective concurrently, both strictly before %b. This
// checks that distributed-reorder-schedule actually moves %c ahead of %b to
// match that admission order, rather than leaving textual order unchanged.
// CHECK-LABEL: module @two_kernels_interleave_collective
// CHECK: %[[A:.*]] = stablehlo.add %arg0, %arg0
// CHECK: %[[START:.*]] = "stablehlo.async_start"(%[[A]])
// CHECK: %[[DONE:.*]] = "stablehlo.async_done"(%[[START]])
// CHECK: %[[C:.*]] = stablehlo.subtract %arg1, %arg1
// CHECK: %[[B:.*]] = stablehlo.multiply %[[DONE]], %[[DONE]]
// CHECK: return
module @two_kernels_interleave_collective {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
    %a = stablehlo.add %arg0, %arg0 : tensor<4xf32>

    %start = "stablehlo.async_start"(%a) ({
    ^bb0(%x: tensor<4xf32>):
      %r = "stablehlo.all_reduce"(%x) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
      ^bb1(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }) : (tensor<4xf32>) -> tensor<4xf32>
      stablehlo.return %r : tensor<4xf32>
    }) {distributed.primitive_cost = {kind = "all-reduce", payload_in = 16 : i64, payload_out = 16 : i64, latency = 1.0 : f64, transfer_time = 20.0 : f64, isolated_duration = 21.0 : f64, volume = [16.0], rho = [1.0]}} : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
    %done = "stablehlo.async_done"(%start) : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>

    %b = stablehlo.multiply %done, %done : tensor<4xf32>
    %c = stablehlo.subtract %arg1, %arg1 : tensor<4xf32>
    return
  }
}

// -----

// A block with no anchor ops (no compute, no collective, no call): the pass
// finds nothing to reorder and leaves the block exactly as it was.
// CHECK-LABEL: module @no_anchors
// CHECK: func.func @main
// CHECK-NEXT: return
module @no_anchors {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main() {
    return
  }
}

// -----

// A single collective with no compute to overlap: nothing to reorder around
// it, so the async_start/async_done pair (and its data producer) stay in
// their original relative order.
// CHECK-LABEL: module @lone_collective
// CHECK: %[[IN:.*]] = stablehlo.add %arg0, %arg0
// CHECK: %[[START:.*]] = "stablehlo.async_start"(%[[IN]])
// CHECK: "stablehlo.async_done"(%[[START]])
// CHECK: return
module @lone_collective {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%arg0: tensor<4xf32>) {
    %in = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    %start = "stablehlo.async_start"(%in) ({
    ^bb0(%x: tensor<4xf32>):
      %r = "stablehlo.all_reduce"(%x) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
      ^bb1(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }) : (tensor<4xf32>) -> tensor<4xf32>
      stablehlo.return %r : tensor<4xf32>
    }) {distributed.primitive_cost = {kind = "all-reduce", payload_in = 16 : i64, payload_out = 16 : i64, latency = 1.0 : f64, transfer_time = 20.0 : f64, isolated_duration = 21.0 : f64, volume = [16.0], rho = [1.0]}} : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
    %done = "stablehlo.async_done"(%start) : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>
    return
  }
}

// -----

// A call's callee body is reordered too, recursively, via its own fresh
// simulation. ScheduleGraph.h's FunctionCall barrier handles any
// CallOpInterface, so a plain func.call exercises the same recursion path
// as distributed.DistributedCall with simpler syntax. The callee here has
// the same independent-op-overlaps-collective shape as the first test
// above, entirely inside the callee's own block.
// CHECK-LABEL: module @call_recursion
// CHECK: call @callee
// CHECK-LABEL: func.func private @callee
// CHECK: %[[A:.*]] = stablehlo.add %arg0, %arg0
// CHECK: %[[START:.*]] = "stablehlo.async_start"(%[[A]])
// CHECK: %[[DONE:.*]] = "stablehlo.async_done"(%[[START]])
// CHECK: %[[C:.*]] = stablehlo.subtract %arg1, %arg1
// CHECK: %[[B:.*]] = stablehlo.multiply %[[DONE]], %[[DONE]]
module @call_recursion {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]

  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
    func.call @callee(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> ()
    return
  }

  func.func private @callee(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
    %a = stablehlo.add %arg0, %arg0 : tensor<4xf32>

    %start = "stablehlo.async_start"(%a) ({
    ^bb0(%x: tensor<4xf32>):
      %r = "stablehlo.all_reduce"(%x) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
      ^bb1(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }) : (tensor<4xf32>) -> tensor<4xf32>
      stablehlo.return %r : tensor<4xf32>
    }) {distributed.primitive_cost = {kind = "all-reduce", payload_in = 16 : i64, payload_out = 16 : i64, latency = 1.0 : f64, transfer_time = 20.0 : f64, isolated_duration = 21.0 : f64, volume = [16.0], rho = [1.0]}} : (tensor<4xf32>) -> !stablehlo.future<tensor<4xf32>>
    %done = "stablehlo.async_done"(%start) : (!stablehlo.future<tensor<4xf32>>) -> tensor<4xf32>

    %b = stablehlo.multiply %done, %done : tensor<4xf32>
    %c = stablehlo.subtract %arg1, %arg1 : tensor<4xf32>
    return
  }
}
