// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics --distributed-annotate-kernel-latency %s | FileCheck %s

// The roofline backend (registered as "cpu") annotates every top-level op of
// a kernel body with its roofline time. The mesh gives F = 1 FLOP/ns and
// M = 10 bytes/ns:
//   - dot_general 2x4 . 4x3: 2 * 6 * 4 = 48 flops, (8 + 12 + 6) * 4 = 104
//     bytes, so max(48 / 1, 104 / 10) = 48ns;
//   - add on 2x3: 6 flops, 3 * 24 = 72 bytes, so max(6, 7.2) = 7.2ns;
//   - reshape is free: 0ns.
// CHECK-LABEL: func.func @roofline_latency
// CHECK: distributed.DistributedKernel
// CHECK: stablehlo.dot_general {{.*}} {distributed.latency = 4.800000e+01 : f64}
// CHECK: stablehlo.add {{.*}} {distributed.latency = 7.200000e+00 : f64}
// CHECK: stablehlo.reshape {{.*}} {distributed.latency = 0.000000e+00 : f64}
// CHECK: distributed.DistributedYield
distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e9, device_mem_bandwidth = 1.0e10}
func.func @roofline_latency(%x: tensor<2x4xf32>, %w: tensor<4x3xf32>) -> tensor<3x2xf32> {
  %r = distributed.DistributedKernel (%x : tensor<2x4xf32>, %w : tensor<4x3xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      -> (tensor<3x2xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[], []] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<2x4xf32>, %b: tensor<4x3xf32>):
    %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %s = stablehlo.add %d, %d : tensor<2x3xf32>
    %t = stablehlo.reshape %s : (tensor<2x3xf32>) -> tensor<3x2xf32>
    distributed.DistributedYield (%t : tensor<3x2xf32>)
  }
  return %r : tensor<3x2xf32>
}

// -----

// A device_target with no registered cost backend is diagnosed, naming the
// registered ones.
// expected-error @below {{no cost backend registered for device_target 'gpu' (registered: cpu, roofline)}}
distributed.PhysicalMesh @mesh0 device_target "gpu" axes [!distributed.physical_comm_axis<2, 1>]
func.func @negative_unregistered_target(%x: tensor<4xf32>) -> tensor<4xf32> {
  %r = distributed.DistributedKernel (%x : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<4xf32>):
    %s = stablehlo.add %a, %a : tensor<4xf32>
    distributed.DistributedYield (%s : tensor<4xf32>)
  }
  return %r : tensor<4xf32>
}
