// -- Smoke tests and IR dump checks for the distributed pass chain.
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --sdy-drop-sharding-and-mesh --convert-main-to-distributed-function --materialize-distributed-collectives --cluster-distributed-kernels="dump-logical-axes=true" -split-input-file -o /dev/null %s 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --sdy-drop-sharding-and-mesh --convert-main-to-distributed-function --materialize-distributed-collectives --cluster-distributed-kernels="dump-logical-axes=true" --mlir-print-ir-after=cluster-distributed-kernels -split-input-file -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHAIN
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --sdy-drop-sharding-and-mesh --convert-main-to-distributed-function --mlir-print-ir-after=convert-main-to-distributed-function -split-input-file -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CONVERT
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --sdy-drop-sharding-and-mesh --convert-main-to-distributed-function --materialize-distributed-collectives --mlir-print-ir-after=materialize-distributed-collectives -split-input-file -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=MATERIALIZE

// -- Pipeline equivalence: the bundled pipeline must match the explicit pass chain.
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline --canonicalize --stabilize-axis-order -split-input-file %s -o - > %t.pipeline
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --sdy-drop-sharding-and-mesh --convert-main-to-distributed-function --materialize-distributed-collectives --cluster-distributed-kernels --canonicalize --stabilize-axis-order -split-input-file %s -o - > %t.explicit
// RUN: diff -u %t.explicit %t.pipeline

// -- Stage-by-stage equivalence: pipeline must match round-tripping to file
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --sdy-drop-sharding-and-mesh -split-input-file %s -o %t.stage1
// RUN: enzymexlamlir-opt --convert-main-to-distributed-function -split-input-file %t.stage1 -o %t.stage2
// RUN: enzymexlamlir-opt --materialize-distributed-collectives -split-input-file %t.stage2 -o %t.stage3
// RUN: enzymexlamlir-opt --cluster-distributed-kernels --canonicalize --stabilize-axis-order -split-input-file %t.stage3 -o %t.stage4
// RUN: diff -u %t.stage4 %t.pipeline

// 2D mesh sharding on both X axes.
module @x_plus_xt_2d {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

  func.func @main(
      %x: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
      -> tensor<4x4xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %sum = stablehlo.add %x, %xt : tensor<4x4xf32>
    return %sum : tensor<4x4xf32>
  }
}

// CHECK: op: %{{.*}} = stablehlo.add %{{.*}} : tensor<4x4xf32>
// CHECK-NEXT:     partitioning axes: [a[[A2D:[0-9]+]]:4 ] [a[[B2D:[0-9]+]]:4 ]
// CHECK-NOT: partitioning axes: [a[[A2D]]:4 ] [a[[A2D]]:4 ]
// CONVERT-LABEL: module @x_plus_xt_2d {
// CONVERT-NOT: func.func @main
// CONVERT: distributed.DistributedYield

// -----

// 1D mesh sharding on one X axis only.
module @x_plus_xt_1d {
  sdy.mesh @mesh = <["x"=2]>

  func.func @main(
      %x: tensor<6x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> tensor<6x6xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0] : (tensor<6x6xf32>) -> tensor<6x6xf32>
    %sum = stablehlo.add %x, %xt : tensor<6x6xf32>
    return %sum : tensor<6x6xf32>
  }
}

// CHECK: op: %{{.*}} = stablehlo.add %{{.*}} : tensor<6x6xf32>
// CHECK-NEXT:     partitioning axes: [a[[A1D:[0-9]+]]:6 ] [a[[B1D:[0-9]+]]:6 ]
// CHECK-NOT: partitioning axes: [a[[A1D]]:6 ] [a[[A1D]]:6 ]

// -----

// No sharding on either X axis.
module @x_plus_xt_none {
  sdy.mesh @mesh = <["x"=2]>

  func.func @main(
      %x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
      -> tensor<8x8xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %sum = stablehlo.add %x, %xt : tensor<8x8xf32>
    return %sum : tensor<8x8xf32>
  }
}

// CHECK: op: %{{.*}} = stablehlo.add %{{.*}} : tensor<8x8xf32>
// CHECK-NEXT:     partitioning axes: [a[[ANONE:[0-9]+]]:8 ] [a[[ANONE:[0-9]+]]:8 ]
// CHECK-NOT: partitioning axes: [a[[ANONE]]:8 ] [a[[ANONE]]:8 ]

// -----

// Reduce over a sharded axis; the resulting collective should carry a reduction
// group over the mesh axis instead of just a pointwise pass-through.
module @reduce_over_sharded_axis {
  sdy.mesh @mesh = <["x"=2]>

  func.func @main(
      %x: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    return %sum : tensor<8xf32>
  }
}

// CHAIN-LABEL: module @reduce_over_sharded_axis {
// CHAIN: distributed.Collective
// CHAIN-SAME: reduces (%{{.*}})
// CHAIN-SAME: maps %{{.*}} : !axis.map

// MATERIALIZE-LABEL: module @reduce_over_sharded_axis {
// MATERIALIZE: distributed.Collective
// MATERIALIZE-SAME: reduces (%{{.*}})
// MATERIALIZE-SAME: maps %{{.*}} : !axis.map
// MATERIALIZE-NOT: distributed.DistributedKernel

// -----

// Reduce over one sharded axis, then consume the reduced value through both a
// same-layout use and a conflicting-layout transpose path. This should
// materialize one reduction collective and one chained layout collective.
module @reduce_then_conflict_chain {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

  func.func @main(
      %x: tensor<8x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {"x"}]>})
      -> tensor<8x8xf32> {
    %cst = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x8x4xf32>, tensor<f32>) -> tensor<8x8xf32>
    %red_t = stablehlo.transpose %red, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %sum = stablehlo.add %red, %red_t : tensor<8x8xf32>
    return %sum : tensor<8x8xf32>
  }
}

// CHAIN-LABEL: module @reduce_then_conflict_chain {
// CHAIN: distributed.Collective
// CHAIN-SAME: reduces (%{{.*}})
// CHAIN-SAME: maps %{{.*}} : !axis.map
// CHAIN: distributed.Await
// CHAIN: distributed.Collective
// CHAIN-SAME: reduces ()
// CHAIN-SAME: maps %{{.*}} : !axis.map

// MATERIALIZE-LABEL: module @reduce_then_conflict_chain {
// MATERIALIZE: distributed.Collective
// MATERIALIZE-SAME: reduces (%{{.*}})
// MATERIALIZE-SAME: maps %{{.*}} : !axis.map
// MATERIALIZE: distributed.Await
// MATERIALIZE: distributed.Collective
// MATERIALIZE-SAME: reduces ()
// MATERIALIZE-SAME: maps %{{.*}} : !axis.map
// MATERIALIZE-NOT: distributed.DistributedKernel

// -----

// Elementwise chain from X merged with Y, ending in a reduce, plus an
// independent elementwise on Y that is also returned. This checks that
// clustering can walk backwards to include Y via X+Y and then forwards to
// include Y-only consumers, yielding multiple outputs from one kernel.
module @test_single_kernel {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

  func.func @main(
      %x: tensor<8x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {"x"}]>},
      %y: tensor<8x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {"x"}]>})
      -> (tensor<8x8xf32>, tensor<8x8x4xf32>) {
    %one = stablehlo.constant dense<1.0> : tensor<8x8x4xf32>
    %x_bias = stablehlo.add %x, %one : tensor<8x8x4xf32>
    %xy = stablehlo.add %x_bias, %y : tensor<8x8x4xf32>
    %y_elem = stablehlo.multiply %y, %one : tensor<8x8x4xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%xy init: %zero) applies stablehlo.add across dimensions = [2] : (tensor<8x8x4xf32>, tensor<f32>) -> tensor<8x8xf32>
    return %red, %y_elem : tensor<8x8xf32>, tensor<8x8x4xf32>
  }
}

// CHAIN-LABEL: module @test_single_kernel {
// CHAIN: %[[KERNEL:.*]]:2 = distributed.DistributedKernel
// CHAIN: stablehlo.add
// CHAIN: stablehlo.add
// CHAIN: stablehlo.multiply
// CHAIN: stablehlo.reduce
// CHAIN: distributed.DistributedYield
// CHAIN: %[[COLL:.*]] = distributed.Collective
// CHAIN-SAME: reduces (%{{.*}})
// CHAIN: %[[RED_AWAIT:.*]] = distributed.Await %[[COLL]]
// CHAIN: builtin.unrealized_conversion_cast %[[RED_AWAIT]]
// CHAIN: distributed.DistributedYield %{{.*}}, %{{.*}}
