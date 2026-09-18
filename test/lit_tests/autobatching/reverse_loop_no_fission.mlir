// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @reverse_step(%tape: tensor<16x8x8xf64>, %seed: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %cst = stablehlo.constant dense<1.000000e-01> : tensor<8x8xf64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf64>
  %0:2 = stablehlo.while(%iterArg = %c0, %adjoint = %seed) : tensor<i64>, tensor<8x8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %tape, %iterArg, %c0, %c0, sizes = [1, 8, 8] : (tensor<16x8x8xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x8x8xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x8x8xf64>) -> tensor<8x8xf64>
    %4 = stablehlo.multiply %cst, %3 : tensor<8x8xf64>
    %5 = stablehlo.compare GT, %4, %zero : (tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xi1>
    // the adjoint chain: not batchable, and it needs %4 and %5 at full width
    %6 = stablehlo.multiply %adjoint, %4 : tensor<8x8xf64>
    %7 = stablehlo.select %5, %6, %zero : tensor<8x8xi1>, tensor<8x8xf64>
    stablehlo.return %1, %7 : tensor<i64>, tensor<8x8xf64>
  }
  return %0#1 : tensor<8x8xf64>
}

// CHECK-LABEL: func.func @reverse_step
// CHECK-NOT:     -> tensor<16x8x8
// CHECK:         %[[LOOP:.+]]:2 = stablehlo.while(%[[IV:.+]] = %{{.+}}, %[[ADJ:.+]] = %arg1)
// CHECK:         } do {
// CHECK-NEXT:      %[[NEXT:.+]] = stablehlo.add %[[IV]], %{{.+}} : tensor<i64>
// CHECK-NEXT:      %[[ROW:.+]] = stablehlo.dynamic_slice %arg0, %[[IV]], %{{.+}}, %{{.+}}, sizes = [1, 8, 8]
// CHECK-NEXT:      %[[FLAT:.+]] = stablehlo.reshape %[[ROW]]
// CHECK-NEXT:      %[[SCALED:.+]] = stablehlo.multiply %{{.+}}, %[[FLAT]] : tensor<8x8xf64>
// CHECK-NEXT:      %[[MASK:.+]] = stablehlo.compare GT, %[[SCALED]], %{{.+}} : (tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xi1>
// CHECK-NEXT:      %[[MUL:.+]] = stablehlo.multiply %[[ADJ]], %[[SCALED]] : tensor<8x8xf64>
// CHECK-NEXT:      %[[SEL:.+]] = stablehlo.select %[[MASK]], %[[MUL]], %{{.+}}
// CHECK-NEXT:      stablehlo.return %[[NEXT]], %[[SEL]]

// -----

// `subtract` accumulates into a carry only with the carry on the left:
// `x - a - b` sums the negations, `a - x` alternates. So this loop's carry does
// not collapse into a reduce, and the multiply feeding it must not be batched
// either -- otherwise the 16x8x8 buffer is materialized and the loop survives
// to read it back, which is the blowup with none of the payoff.

func.func @carry_on_rhs(%tape: tensor<16x8x8xf64>, %seed: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %cst = stablehlo.constant dense<1.000000e-01> : tensor<8x8xf64>
  %0:2 = stablehlo.while(%iterArg = %c0, %acc = %seed) : tensor<i64>, tensor<8x8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %tape, %iterArg, %c0, %c0, sizes = [1, 8, 8] : (tensor<16x8x8xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x8x8xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x8x8xf64>) -> tensor<8x8xf64>
    %4 = stablehlo.multiply %cst, %3 : tensor<8x8xf64>
    %5 = stablehlo.subtract %4, %acc : tensor<8x8xf64>
    stablehlo.return %1, %5 : tensor<i64>, tensor<8x8xf64>
  }
  return %0#1 : tensor<8x8xf64>
}

// CHECK-LABEL: func.func @carry_on_rhs
// CHECK-NOT:     -> tensor<16x8x8
// CHECK:         stablehlo.while
// CHECK:         } do {
// CHECK-NEXT:      %[[NEXT:.+]] = stablehlo.add
// CHECK-NEXT:      %[[ROW:.+]] = stablehlo.dynamic_slice %arg0
// CHECK-NEXT:      %[[FLAT:.+]] = stablehlo.reshape %[[ROW]]
// CHECK-NEXT:      %[[MUL:.+]] = stablehlo.multiply %{{.+}}, %[[FLAT]] : tensor<8x8xf64>
// CHECK-NEXT:      %[[SUB:.+]] = stablehlo.subtract %[[MUL]], %{{.+}} : tensor<8x8xf64>
// CHECK-NEXT:      stablehlo.return %[[NEXT]], %[[SUB]]
