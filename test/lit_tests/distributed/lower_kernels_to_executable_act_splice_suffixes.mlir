// RUN: enzymexlamlir-opt %s --distributed-lower-kernels-to-executable="act-patched-module-file=%S/act_patched_two_launch_kernel.mlir.txt" | FileCheck %s

// ACT names a second launch's declarations with a `_1` suffix, the same shape
// SymbolTable's uniquing appends on a collision. The module already holds the
// first launch's tagged names, so splicing renames those two siblings on
// insertion; neither renamed name may coincide with a sibling still to be
// spliced (@act_stage_0_1), or the rewrite of that sibling's calls would
// rebind the first launch's stage to the second's declaration. The fixture's
// stages take different tensor types, so such a mix-up would also fail the
// verifier.

// CHECK-LABEL: func.func @one_kernel
// CHECK: distributed.DistributedKernel
// CHECK: call @act_stage_0_actcall_[[S0:[0-9]+]](%arg2, %{{.*}}) : (tensor<4x8xf32>
// CHECK-NEXT: call @act_launch_actcall_[[L0:[0-9]+]](
// CHECK-NEXT: func.call @act_fetch_actcall(
// CHECK-NEXT: call @act_stage_0_1_actcall(%arg3, %{{.*}}) : (tensor<8x4xf32>
// CHECK-NEXT: call @act_launch_1_actcall(
// CHECK-NEXT: func.call @act_fetch_1_actcall(

// CHECK: func.func private @act_launch_actcall_[[L0]]({{.*}}act.kernel = "first"
// CHECK: func.func private @act_stage_0_actcall_[[S0]](tensor<4x8xf32>
// CHECK: func.func private @act_fetch_actcall(
// CHECK: func.func private @act_launch_1_actcall({{.*}}act.kernel = "second"
// CHECK: func.func private @act_stage_0_1_actcall(tensor<8x4xf32>
// CHECK: func.func private @act_fetch_1_actcall(

func.func private @act_launch_actcall()
func.func private @act_stage_0_actcall()

func.func @one_kernel(%lhs: tensor<4x8xf32>, %rhs: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %r = distributed.DistributedKernel (%lhs : tensor<4x8xf32>, %rhs : tensor<8x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<4x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>):
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    distributed.DistributedYield (%d : tensor<4x4xf32>)
  }
  return %r : tensor<4x4xf32>
}
