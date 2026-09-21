// RUN: enzymexlamlir-opt --verify-diagnostics --distributed-print-compute-cost %s -o /dev/null
// RUN: enzymexlamlir-opt --distributed-print-compute-cost="flops=1.0e6 mem-bandwidth=1.0e6" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=FAST

// distributed-print-compute-cost reports the per-op roofline of the ops in
// each distributed kernel body: time = max(flops / F, bytes / M), with bytes
// the operand and result bytes (ComputeCost.h). The mesh below gives F = 100
// FLOP per time unit and M = 100 bytes per time unit.
//
//   dot_general 8x16 . 16x4 -> 8x4, f32:
//     flops = 2 * 32 * 16 = 1024, bytes = (128 + 64 + 32) * 4 = 896
//     time = max(10.24, 8.96) = 10.24, compute bound
//   add of two 8x4 f32 -> 8x4 f32:
//     flops = 32, bytes = 3 * 128 = 384, time = max(0.32, 3.84) = 3.84,
//     memory bound
//   compare of two 8x4 f32 -> 8x4 i1 (1 byte per element):
//     flops = 32, bytes = 2 * 128 + 32 = 288, time = 2.88
//   exponential 8x4 f32: a transcendental, weight 1 per element:
//     flops = 32, bytes = 256, time = 2.56
//   reshape: free, so it is costed at zero and not reported as unknown
//   count_leading_zeros 4 x i32: not in the table, so bytes only:
//     flops = 0, bytes = 2 * 16 = 32, time = 0.32, counted as unknown
//   the DistributedYield terminator is not costed
//
// Total = 10.24 + 3.84 + 2.88 + 2.56 + 0 + 0.32 = 19.84 over 6 ops.
// With the flops and mem-bandwidth options set to 1e6, which override the
// mesh, every time shrinks by 1e4 (0.001984 in total).
// FAST: compute cost: total=0.001984

// expected-remark @below {{compute cost: total=19.84 over 1 kernels, 6 ops (unknown: 1, collective: 0, skipped: 0)}}
module {
  distributed.PhysicalMesh @mesh device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e2, device_mem_bandwidth = 1.0e2}

  func.func @main(%a: tensor<8x16xf32>, %b: tensor<16x4xf32>, %i: tensor<4xi32>) -> (tensor<32xf32>, tensor<4xi32>) {
    %r:2 = distributed.DistributedKernel (%a : tensor<8x16xf32>, %b : tensor<16x4xf32>, %i : tensor<4xi32>) <[<dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[], []] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<32xf32>, tensor<4xi32>) <[<dim_partitioning_axes = [[]] : unreduced_axes = []>, <dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
    ^bb0(%x: tensor<8x16xf32>, %y: tensor<16x4xf32>, %z: tensor<4xi32>):
      // expected-remark @below {{flops=1024 bytes=896 time=10.24}}
      %d = stablehlo.dot_general %x, %y, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
      // expected-remark @below {{flops=32 bytes=384 time=3.84}}
      %s = stablehlo.add %d, %d : tensor<8x4xf32>
      // expected-remark @below {{flops=32 bytes=288 time=2.88}}
      %c = stablehlo.compare GT, %s, %d : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
      // expected-remark @below {{flops=32 bytes=256 time=2.56}}
      %e = stablehlo.exponential %s : tensor<8x4xf32>
      // expected-remark @below {{flops=0 bytes=0 time=0}}
      %f = stablehlo.reshape %e : (tensor<8x4xf32>) -> tensor<32xf32>
      // expected-remark @below {{flops=0 bytes=32 time=0.32 (unknown op)}}
      %n = stablehlo.count_leading_zeros %z : tensor<4xi32>
      distributed.DistributedYield (%f : tensor<32xf32>, %n : tensor<4xi32>)
    }
    return %r#0, %r#1 : tensor<32xf32>, tensor<4xi32>
  }
}
