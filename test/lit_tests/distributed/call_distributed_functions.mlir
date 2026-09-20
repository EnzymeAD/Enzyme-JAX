// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s

// A call is localized: its operands and results are per-device shards, bound
// to the callee's axes by casts to and from the global neighbors.
// A shared callee becomes a distributed function of its own: its body is
// clustered into kernels once, and each call site is a distributed call that
// stays outside every kernel.
// CHECK-NOT: func.func
// CHECK-NOT: func.call
// CHECK: "distributed.DistributedFunction"
// CHECK-SAME: sym_name = "main"
// CHECK-NOT: distributed.DistributedKernel
// CHECK: distributed.CastGlobalToLocal
// CHECK: distributed.DistributedCall @mlp 
// CHECK-SAME: tensor<1xf32>
// CHECK: distributed.CastLocalToGlobal
// CHECK-NOT: distributed.DistributedKernel
// CHECK: distributed.DistributedCall @mlp 
// CHECK-NOT: distributed.DistributedKernel
// CHECK: "distributed.DistributedFunction"
// CHECK-SAME: sym_name = "mlp"
// CHECK: distributed.DistributedKernel
module @calls {
  sdy.mesh @mesh = <["x"=2]>

  func.func private @mlp(%h: tensor<8xf32>, %wa: tensor<16x8xf32>,
                         %wb: tensor<8x16xf32>) -> tensor<8xf32> {
    %t = stablehlo.dot_general %wa, %h, contracting_dims = [1] x [0]
        : (tensor<16x8xf32>, tensor<8xf32>) -> tensor<16xf32>
    %r = stablehlo.dot_general %wb, %t, contracting_dims = [1] x [0]
        : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }

  func.func @main(
      %x: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      %wa1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %wb1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %wa2: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %wb2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> tensor<8xf32> {
    %a = func.call @mlp(%x, %wa1, %wb1)
        : (tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<8xf32>
    %b = func.call @mlp(%a, %wa2, %wb2)
        : (tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<8xf32>
    return %b : tensor<8xf32>
  }
}
