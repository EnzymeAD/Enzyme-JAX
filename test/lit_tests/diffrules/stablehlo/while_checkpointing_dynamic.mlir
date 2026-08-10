// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

// Periodic checkpointing of a loop whose trip count is only known at runtime:
// the limit is a function argument, so neither the number of segments nor the
// length of the last one can be folded. The period has to be given, since
// there is no compile-time N to take the square root of.
//
// opt + FileCheck only, no interpretation: a dynamic trip count gives the cache
// a dynamic shape (it does so for a plain reverse loop too, not just a
// checkpointed one), and the StableHLO interpreter rejects dynamic result
// types. The static-trip-count tests next to this one are the ones that pin the
// numerics of the same formulas.

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
    %n9 = stablehlo.constant dense<9> : tensor<i64>

    %ckpt10:2 = enzyme.autodiff @with_checkpointing(%input, %n10, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %plain10:2 = enzyme.autodiff @without_checkpointing(%input, %n10, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)


    %ckpt9:2 = enzyme.autodiff @with_checkpointing(%input, %n9, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %plain9:2 = enzyme.autodiff @without_checkpointing(%input, %n9, %diffe) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)


    return
  }
}

// CHECK-LABEL: func.func private @diffewith_checkpointing

// The segment count is ceil(numIters / 3), computed at runtime.
// CHECK:         stablehlo.divide

// Forward: an outer segment loop whose body recomputes a segment of at most 3
// iterations, the length clamped at runtime because the last one may be short.
// CHECK:         stablehlo.while
// CHECK:           stablehlo.minimum
// CHECK:           stablehlo.while

// Reverse: the segments are replayed back to front, again with the clamp.
// CHECK:         stablehlo.while
// CHECK:           stablehlo.subtract
// CHECK:           stablehlo.minimum
// CHECK:           stablehlo.while
