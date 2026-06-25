// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=loop_unswitch" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A while loop whose body contains an if with a loop-invariant predicate should
// be split into two specialised while loops wrapped in an outer if.

// CHECK-LABEL: func.func @versioning_basic
// CHECK-NEXT:    %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[C2:.*]] = stablehlo.constant dense<2> : tensor<i64>
// Outer if on the invariant predicate (%arg0).
// CHECK-NEXT:    %[[OUTER:.*]]:2 = "stablehlo.if"(%arg0) ({
// True branch: while body inlines the true branch of the original if (step = %c_0 = 1).
// CHECK-NEXT:      %[[TW:.*]]:2 = stablehlo.while(%[[TA:.*]] = %[[C0]], %[[TB:.*]] = %[[C0]])
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %[[TCMP:.*]] = stablehlo.compare LT, %[[TA]], %arg1
// CHECK-NEXT:        stablehlo.return %[[TCMP]]
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %[[TI:.*]] = stablehlo.add %[[TA]], %[[C1]]
// CHECK-NEXT:        %[[TS:.*]] = stablehlo.add %[[TB]], %[[C1]]
// CHECK-NEXT:        stablehlo.return %[[TI]], %[[TS]]
// CHECK-NEXT:      }
// CHECK-NEXT:      stablehlo.return %[[TW]]#0, %[[TW]]#1
// False branch: while body inlines the false branch of the original if (step = %c_1 = 2).
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %[[FW:.*]]:2 = stablehlo.while(%[[FA:.*]] = %[[C0]], %[[FB:.*]] = %[[C0]])
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %[[FCMP:.*]] = stablehlo.compare LT, %[[FA]], %arg1
// CHECK-NEXT:        stablehlo.return %[[FCMP]]
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %[[FI:.*]] = stablehlo.add %[[FA]], %[[C1]]
// CHECK-NEXT:        %[[FS:.*]] = stablehlo.add %[[FB]], %[[C2]]
// CHECK-NEXT:        stablehlo.return %[[FI]], %[[FS]]
// CHECK-NEXT:      }
// CHECK-NEXT:      stablehlo.return %[[FW]]#0, %[[FW]]#1
// CHECK-NEXT:    }) : (tensor<i1>) -> (tensor<i64>, tensor<i64>)
// CHECK-NEXT:    return %[[OUTER]]#1
func.func @versioning_basic(%pred: tensor<i1>, %limit: tensor<i64>) -> tensor<i64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %result:2 = stablehlo.while(%iterArg = %c0, %iterArg_1 = %c0) : tensor<i64>, tensor<i64>
   cond {
    %cmp = stablehlo.compare  LT, %iterArg, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %step = "stablehlo.if"(%pred) ({
      stablehlo.return %c1 : tensor<i64>
    }, {
      stablehlo.return %c2 : tensor<i64>
    }) : (tensor<i1>) -> tensor<i64>
    %new_i = stablehlo.add %iterArg, %c1 : tensor<i64>
    %new_sum = stablehlo.add %iterArg_1, %step : tensor<i64>
    stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }
  return %result#1 : tensor<i64>
}

// When the if predicate is defined inside the loop (not loop-invariant),
// the pattern must not fire — the while and the if inside it are preserved.

// CHECK-LABEL: func.func @no_versioning_variant_pred
// CHECK:         stablehlo.while
// CHECK:         "stablehlo.if"
// CHECK-NOT:     "stablehlo.if"(%arg{{[^,)]*}}) ({
func.func @no_versioning_variant_pred(%limit: tensor<i64>) -> tensor<i64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %result:2 = stablehlo.while(%iterArg = %c0, %iterArg_1 = %c0) : tensor<i64>, tensor<i64>
   cond {
    %cmp = stablehlo.compare  LT, %iterArg, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    // Predicate computed from the loop-variant iter arg — NOT invariant.
    %variant_pred = stablehlo.compare  EQ, %iterArg, %c0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %step = "stablehlo.if"(%variant_pred) ({
      stablehlo.return %c1 : tensor<i64>
    }, {
      stablehlo.return %c2 : tensor<i64>
    }) : (tensor<i1>) -> tensor<i64>
    %new_i = stablehlo.add %iterArg, %c1 : tensor<i64>
    %new_sum = stablehlo.add %iterArg_1, %step : tensor<i64>
    stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }
  return %result#1 : tensor<i64>
}
