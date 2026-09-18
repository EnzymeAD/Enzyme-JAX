// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt %s --distributed-lower-kernels-to-executable="dump-kernel-modules-to=%t.dir" | FileCheck %s --check-prefix=UNCHANGED
// RUN: ls %t.dir | wc -l | FileCheck %s --check-prefix=COUNT2
// RUN: cat %t.dir/kernel-*.mlir | FileCheck %s --check-prefix=DUMPS

// This pass runs once per search candidate (it sits at the end of
// buildDistributedSearchLoweringPipeline), sharing one dump directory
// across many invocations, so re-running it into the same directory must
// add dumps rather than overwrite the previous run's -- each kernel gets a
// randomly-suffixed name rather than one keyed on a per-invocation index.
// RUN: enzymexlamlir-opt %s --distributed-lower-kernels-to-executable="dump-kernel-modules-to=%t.dir" | FileCheck %s --check-prefix=UNCHANGED
// RUN: ls %t.dir | wc -l | FileCheck %s --check-prefix=COUNT4

// COUNT2: 2
// COUNT4: 4

// The pass only walks the module -- it never rewrites it -- so both kernels
// survive completely unchanged when a dump directory is requested.
// UNCHANGED-LABEL: func.func @plain
// UNCHANGED: distributed.DistributedKernel (%arg0 : tensor<4xf32>)
// UNCHANGED-LABEL: func.func @captured_constant
// UNCHANGED: distributed.DistributedKernel (%arg0 : tensor<4xf32>)

// Dump file names are random, so their content is checked in whatever
// order `cat`'s glob happens to list them in (DUMPS-DAG), rather than
// relying on file identity.
// DUMPS-DAG: stablehlo.add %arg0, %arg0
func.func @plain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %r0 = distributed.DistributedKernel (%arg0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<4xf32>):
    %c = stablehlo.add %a, %a : tensor<4xf32>
    distributed.DistributedYield (%c : tensor<4xf32>)
  }
  return %r0 : tensor<4xf32>
}

// A constant referenced directly inside a kernel body without being one of
// the kernel's own operands/block-args -- e.g. what CSE leaves behind when
// it commons a constant shared by several sibling kernels and hoists the
// single copy outside all of them. The dumped module must be fully
// self-contained, so this constant should be cloned into it (DUMPS-NOT
// below rules out the dangling reference this used to produce) rather than
// left referring to a value that doesn't exist there.
// DUMPS-DAG: stablehlo.constant dense<0.000000e+00>
// DUMPS-DAG: stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0]
// DUMPS-NOT: UNKNOWN SSA VALUE
func.func @captured_constant(%arg0: tensor<4xf32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %r0 = distributed.DistributedKernel (%arg0 : tensor<4xf32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [[]] : unreduced_axes = []>]>
      -> (tensor<f32>) #distributed.indexed_tensor_sharding_per_value<[<dim_partitioning_axes = [] : unreduced_axes = []>]>
      axes () {
  ^bb0(%a: tensor<4xf32>):
    %s = stablehlo.reduce(%a init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    distributed.DistributedYield (%s : tensor<f32>)
  }
  return %r0 : tensor<f32>
}
