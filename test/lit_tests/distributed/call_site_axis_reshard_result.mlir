// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --convert-main-to-distributed-function="dump-value-axes=true" -o /dev/null %s 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --convert-main-to-distributed-function="dump-value-axes=true" -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=SEPARATE

// A callee that returns a sdy.reshard result. The reshard breaks the propagation
// dependency, so the call results carry the reshard's output axes rather than
// the callee argument's; both call sites share those output axes, and they unify
// with what consumes the results.
//
// The operands %arg0/%arg1 share the callee argument's axes, and the results
// and everything consuming them share the reshard's, which must differ.
// CHECK: ===== Logical Axes per Value =====
// CHECK-NEXT: %arg0 : {{\[}}[[IN0:a[0-9]+]]{{\]}} {{\[}}[[IN1:a[0-9]+]]{{\]}}
// CHECK-NEXT: %arg1 : {{\[}}[[IN0]]{{\]}} {{\[}}[[IN1]]{{\]}}
// CHECK-NEXT: %arg2 : {{\[}}[[OUT0:a[0-9]+]]{{\]}} {{\[}}[[OUT1:a[0-9]+]]{{\]}}
// CHECK-NEXT: %0 : {{\[}}[[OUT0]]{{\]}} {{\[}}[[OUT1]]{{\]}}
// CHECK-NEXT: %1 : {{\[}}[[OUT0]]{{\]}} {{\[}}[[OUT1]]{{\]}}
// CHECK-NEXT: %2 : {{\[}}[[OUT0]]{{\]}} {{\[}}[[OUT1]]{{\]}}
// CHECK-NEXT: %3 : {{\[}}[[OUT0]]{{\]}} {{\[}}[[OUT1]]{{\]}}
//
// FileCheck cannot assert that two captures differ, so the SEPARATE run instead
// captures the operand axes and requires that they appear in none of the later
// values.
// SEPARATE: %arg1 : {{\[}}[[IN0:a[0-9]+]]{{\]}} {{\[}}[[IN1:a[0-9]+]]{{\]}}
// SEPARATE-NOT: {{\[}}[[IN0]]{{\]}}
// SEPARATE-NOT: {{\[}}[[IN1]]{{\]}}
// SEPARATE: ===== End Logical Axes per Value =====
module @reshard_result {
  sdy.mesh @mesh = <["x"=2]>

  func.func private @re(
      %h: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> tensor<8x8xf32> {
    %r = sdy.reshard %h <@mesh, [{}, {"x"}]> : tensor<8x8xf32>
    return %r : tensor<8x8xf32>
  }

  func.func @main(
      %x: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %y: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %w: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> tensor<8x8xf32> {
    %a = func.call @re(%x) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %b = func.call @re(%y) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %s = stablehlo.add %a, %w : tensor<8x8xf32>
    %t = stablehlo.add %b, %s : tensor<8x8xf32>
    return %t : tensor<8x8xf32>
  }
}
