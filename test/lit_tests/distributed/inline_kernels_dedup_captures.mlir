// RUN: enzymexlamlir-opt --distributed-annotate-kernel-latency --distributed-inline-kernels --cse %s | FileCheck %s

// Kernel costing splices back a module that carries its own copy of each
// value a kernel captures from outside (buildKernelBodyModule), so a constant
// hoisted above two kernels ends up copied into both. Once the kernels are
// inlined, CSE, which distributed-search-lowering-pipeline runs right after
// distributed-inline-kernels, merges the copies and erases the now-unused
// outer constant, leaving one.
// CHECK-LABEL: func.func @hoisted_constant_two_kernels
// CHECK: stablehlo.constant
// CHECK-NOT: stablehlo.constant
// CHECK: return
distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
func.func @hoisted_constant_two_kernels(%x: tensor<4xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
  %r0 = distributed.DistributedKernel (%x : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<4xf32>):
    %s = stablehlo.add %a, %cst : tensor<4xf32>
    distributed.DistributedYield (%s : tensor<4xf32>)
  }
  %r1 = distributed.DistributedKernel (%r0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%b: tensor<4xf32>):
    %m = stablehlo.multiply %b, %cst : tensor<4xf32>
    distributed.DistributedYield (%m : tensor<4xf32>)
  }
  return %r1 : tensor<4xf32>
}
