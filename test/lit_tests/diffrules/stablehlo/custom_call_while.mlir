// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=REVERSE

// The custom call runs inside a loop, so its cached primal operands and results
// genuinely have to be pushed once per iteration and popped in reverse -- unlike
// the straight-line tests, where the push/pop pairs cancel out entirely. This is
// what exercises cacheValues' two insertion points: operands before the cloned
// call, results after it.

func.func @scale_rev(%x: tensor<3xf64>, %y: tensor<3xf64>,
                     %dy: tensor<3xf64>) -> tensor<3xf64> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<3xf64>
  %res = stablehlo.multiply %dy, %three : tensor<3xf64>
  func.return %res : tensor<3xf64>
}

func.func @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_10 = stablehlo.constant dense<10> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i64>, tensor<3xf64>
   cond {
    %1 = stablehlo.compare  LT, %iterArg, %c_10,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %3 = stablehlo.custom_call @scale(%iterArg_0) {
      enzyme.reverse = @scale_rev,
      enzyme.active_operands = array<i64: 0>
    } : (tensor<3xf64>) -> tensor<3xf64>
    stablehlo.return %2, %3 : tensor<i64>, tensor<3xf64>
  }
  return %0#1 : tensor<3xf64>
}

// REVERSE: func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// The primal loop still runs the call, and the caches it pushes survive to the
// reverse loop, which calls the rule once per iteration.
// REVERSE:   stablehlo.custom_call @scale(
// REVERSE:   call @scale_rev(
// REVERSE:   return
