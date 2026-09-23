// RUN: enzymexlamlir-opt %s --distributed-lower-kernels-to-executable="act-patched-module-file=%S/act_patched_tiny_kernel.mlir.txt" | FileCheck %s

// spliceKernelBodyModule (Dialect/Distributed/Utilities.cpp) is exercised here
// through --act-patched-module-file, a test-only seam on
// distributed-lower-kernels-to-executable that feeds one literal ACT-shaped
// patched module in place of a real dispatch (not wired up yet). The fixture
// (act_patched_tiny_kernel.mlir.txt, a same-directory sibling read via %S) is
// real ACT output for a standalone kernel doing
// dot_general(tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32> -- the
// same signature both kernels below use, per buildKernelBodyModule's ABI.
//
// Two kernels are spliced from the very same fixture to exercise both cases
// in one pass invocation: the first kernel walked gets ACT's declarations
// (@act_launch/@act_stage_0/@act_stage_1/@act_fetch) inserted under their
// original names, and the second walked kernel collides with those and is
// spliced with the auto-renamed symbols instead -- this is the same collision
// two *different* kernels compiled by ACT would hit in the real pipeline,
// since ACT names these declarations identically for every kernel it
// compiles.

// CHECK-LABEL: func.func @two_kernels

// The first kernel's spliced body: block arguments remapped onto the
// kernel's own (%arg2/%arg3), and the patched func.return became a
// DistributedYieldOp -- calls still reference ACT's original, unrenamed
// declaration names since nothing else in the module claimed them first.
// CHECK: distributed.DistributedKernel
// CHECK: ^bb0(%arg2: tensor<4x8xf32>, %arg3: tensor<8x4xf32>):
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[SV0:.*]] = memref.subview %[[ALLOC0]][0]
// CHECK-NEXT: %[[SV1:.*]] = memref.subview %[[ALLOC0]][128]
// CHECK-NEXT: %[[SV2:.*]] = memref.subview %[[ALLOC0]][256]
// CHECK-NEXT: call @act_stage_0(%arg2, %[[SV0]])
// CHECK-NEXT: call @act_stage_1(%arg3, %[[SV1]])
// CHECK-NEXT: call @act_launch(%[[SV0]], %[[SV1]], %[[SV2]])
// CHECK-NEXT: %[[RES0:.*]] = func.call @act_fetch(%[[SV2]])
// CHECK-NEXT: memref.dealloc %[[ALLOC0]]
// CHECK-NEXT: distributed.DistributedYield (%[[RES0]] : tensor<4x4xf32>)

// The second kernel collides with the first's now-inserted declarations, so
// its own splice's calls were rewritten to the renamed symbols -- not left
// pointing at "act_launch" et al. (the first kernel's, now unrelated to this
// one) and not dangling.
// CHECK: distributed.DistributedKernel
// CHECK: ^bb0(%arg2: tensor<4x8xf32>, %arg3: tensor<8x4xf32>):
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: %[[SV3:.*]] = memref.subview %[[ALLOC1]][0]
// CHECK-NEXT: %[[SV4:.*]] = memref.subview %[[ALLOC1]][128]
// CHECK-NEXT: %[[SV5:.*]] = memref.subview %[[ALLOC1]][256]
// CHECK-NEXT: call @act_stage_0_1(%arg2, %[[SV3]])
// CHECK-NEXT: call @act_stage_1_2(%arg3, %[[SV4]])
// CHECK-NEXT: call @act_launch_0(%[[SV3]], %[[SV4]], %[[SV5]])
// CHECK-NEXT: %[[RES1:.*]] = func.call @act_fetch_3(%[[SV5]])
// CHECK-NEXT: memref.dealloc %[[ALLOC1]]
// CHECK-NEXT: distributed.DistributedYield (%[[RES1]] : tensor<4x4xf32>)

// The first kernel's declarations, spliced in under their original names...
// CHECK: func.func private @act_launch{{.*}}attributes {act.cost = 4.160000e+02 : f64
// CHECK: func.func private @act_stage_0(
// CHECK: func.func private @act_stage_1(
// CHECK: func.func private @act_fetch(
// ...and the second kernel's copies of the same declarations, auto-renamed
// on insertion to avoid colliding with the first's.
// CHECK: func.func private @act_launch_0{{.*}}attributes {act.cost = 4.160000e+02 : f64
// CHECK: func.func private @act_stage_0_1(
// CHECK: func.func private @act_stage_1_2(
// CHECK: func.func private @act_fetch_3(

func.func @two_kernels(%lhs: tensor<4x8xf32>, %rhs: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %r0 = distributed.DistributedKernel (%lhs : tensor<4x8xf32>, %rhs : tensor<8x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<4x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>):
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    distributed.DistributedYield (%d : tensor<4x4xf32>)
  }
  %r1 = distributed.DistributedKernel (%lhs : tensor<4x8xf32>, %rhs : tensor<8x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<4x4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes () {
  ^bb0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>):
    %d = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    distributed.DistributedYield (%d : tensor<4x4xf32>)
  }
  return %r0, %r1 : tensor<4x4xf32>, tensor<4x4xf32>
}
