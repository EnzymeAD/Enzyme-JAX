// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --convert-main-to-distributed-function="dump-value-axes=true" %s 2>&1 | FileCheck %s

// Two call sites of one callee: the callee's argument and result axes are
// shared, so both calls' weights (and the activation flowing through) unify.
// %arg1/%arg3 are the two first-layer weights and %arg2/%arg4 the two
// second-layer weights; each pair must carry identical axes.
// CHECK: ===== Logical Axes per Value =====
// CHECK-NEXT: %arg0 : {{\[}}[[H:a[0-9]+]]{{\]}}
// CHECK-NEXT: %arg1 : {{\[}}[[A:a[0-9]+]]{{\]}} {{\[}}[[H]]{{\]}}
// CHECK-NEXT: %arg2 : {{\[}}[[H]]{{\]}} {{\[}}[[A]]{{\]}}
// CHECK-NEXT: %arg3 : {{\[}}[[A]]{{\]}} {{\[}}[[H]]{{\]}}
// CHECK-NEXT: %arg4 : {{\[}}[[H]]{{\]}} {{\[}}[[A]]{{\]}}
// CHECK-NEXT: %0 : {{\[}}[[H]]{{\]}}
// CHECK-NEXT: %1 : {{\[}}[[H]]{{\]}}
module @two_call_sites {
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
