// RUN: enzymexlamlir-opt %s --enzyme --verify-diagnostics --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize --inline --enzyme-hlo-opt --stablehlo-refine-shapes --tensor-empty-raise | stablehlo-translate --interpret

module {
  func.func private @without_checkpointing(%arg0: tensor<f64>, %n: tensor<i64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  func.func private @with_checkpointing(%arg0: tensor<f64>, %n: tensor<i64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 3 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  func.func @main() {
    %input = stablehlo.constant dense<1.0000001> : tensor<f64>
    %diffe = stablehlo.constant dense<1.0> : tensor<f64>
    %n10 = stablehlo.constant dense<10> : tensor<i64>

    %ckpt:2 = enzyme.autodiff @with_checkpointing(%input, %n10, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %plain:2 = enzyme.autodiff @without_checkpointing(%input, %n10, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    check.expect_almost_eq %ckpt#0, %plain#0 : tensor<f64>
    check.expect_almost_eq %ckpt#1, %plain#1 : tensor<f64>

    return
  }
}

// CHECK-LABEL: func.func private @diffewith_checkpointing

// A segment scaffold would clamp each segment's length against the trip count;
// the plain path has nothing to clamp.
// CHECK-NOT:     stablehlo.minimum

// Forward: one loop, taping into a buffer sized by the trip count itself.
// CHECK:         stablehlo.dynamic_pad
// CHECK:         %[[fwd:.+]]:3 = stablehlo.while
// CHECK-NOT:     stablehlo.minimum

// Reverse: one loop, walking that tape back.
// CHECK:         %[[rev:.+]]:3 = stablehlo.while
// CHECK:           stablehlo.dynamic_slice %[[fwd]]#2
// CHECK-NOT:     stablehlo.minimum
// CHECK:         return %[[fwd]]#1, %[[rev]]#1
