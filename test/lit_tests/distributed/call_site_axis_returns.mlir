// RUN: enzymexlamlir-opt --split-input-file --sdy-propagation-pipeline --sdy-insert-explicit-reshards --convert-main-to-distributed-function="dump-value-axes=true" -o /dev/null %s 2>&1 | FileCheck %s

// How a callee's returned value is traced back to the symbols its call results
// carry: directly a block argument, or another call's result.

// A callee that returns its argument unchanged makes each call result share its
// operand's axes, so the operands of both call sites and everything they feed
// through the adds collapse onto one axis pair.
// CHECK: ===== Logical Axes per Value =====
// CHECK-NEXT: %arg0 : {{\[}}[[P0:a[0-9]+]]{{\]}} {{\[}}[[P1:a[0-9]+]]{{\]}}
// CHECK-NEXT: %arg1 : {{\[}}[[P0]]{{\]}} {{\[}}[[P1]]{{\]}}
// CHECK-NEXT: %arg2 : {{\[}}[[P0]]{{\]}} {{\[}}[[P1]]{{\]}}
// CHECK-NEXT: %0 : {{\[}}[[P0]]{{\]}} {{\[}}[[P1]]{{\]}}
// CHECK-NEXT: %1 : {{\[}}[[P0]]{{\]}} {{\[}}[[P1]]{{\]}}
// CHECK-NEXT: %2 : {{\[}}[[P0]]{{\]}} {{\[}}[[P1]]{{\]}}
// CHECK-NEXT: %3 : {{\[}}[[P0]]{{\]}} {{\[}}[[P1]]{{\]}}
module @passthrough {
  sdy.mesh @mesh = <["x"=2]>

  func.func private @id(%h: tensor<8x8xf32>) -> tensor<8x8xf32> {
    return %h : tensor<8x8xf32>
  }

  func.func @main(
      %x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %y: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %w: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> tensor<8x8xf32> {
    %a = func.call @id(%x) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %b = func.call @id(%y) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %s = stablehlo.add %a, %w : tensor<8x8xf32>
    %t = stablehlo.add %b, %s : tensor<8x8xf32>
    return %t : tensor<8x8xf32>
  }
}

// -----

// A callee that returns another call's result: the outer function's result axes
// are the inner callee's, so this unifies exactly like calling the inner callee
// directly (see call_site_axis_unification.mlir).
// CHECK: ===== Logical Axes per Value =====
// CHECK-NEXT: %arg0 : {{\[}}[[H:a[0-9]+]]{{\]}}
// CHECK-NEXT: %arg1 : {{\[}}[[A:a[0-9]+]]{{\]}} {{\[}}[[H]]{{\]}}
// CHECK-NEXT: %arg2 : {{\[}}[[H]]{{\]}} {{\[}}[[A]]{{\]}}
// CHECK-NEXT: %arg3 : {{\[}}[[A]]{{\]}} {{\[}}[[H]]{{\]}}
// CHECK-NEXT: %arg4 : {{\[}}[[H]]{{\]}} {{\[}}[[A]]{{\]}}
// CHECK-NEXT: %0 : {{\[}}[[H]]{{\]}}
// CHECK-NEXT: %1 : {{\[}}[[H]]{{\]}}
module @nested {
  sdy.mesh @mesh = <["x"=2]>

  func.func private @mlp(%h: tensor<8xf32>, %wa: tensor<16x8xf32>,
                         %wb: tensor<8x16xf32>) -> tensor<8xf32> {
    %t = stablehlo.dot_general %wa, %h, contracting_dims = [1] x [0]
        : (tensor<16x8xf32>, tensor<8xf32>) -> tensor<16xf32>
    %r = stablehlo.dot_general %wb, %t, contracting_dims = [1] x [0]
        : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }

  func.func private @outer(%h: tensor<8xf32>, %wa: tensor<16x8xf32>,
                           %wb: tensor<8x16xf32>) -> tensor<8xf32> {
    %r = func.call @mlp(%h, %wa, %wb)
        : (tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }

  func.func @main(
      %x: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      %wa1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %wb1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %wa2: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %wb2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> tensor<8xf32> {
    %a = func.call @outer(%x, %wa1, %wb1)
        : (tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<8xf32>
    %b = func.call @outer(%a, %wa2, %wb2)
        : (tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<8xf32>
    return %b : tensor<8xf32>
  }
}
