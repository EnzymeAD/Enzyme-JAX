// RUN: enzymexlamlir-opt --lower-enzymexla-math %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s --check-prefix=FOLD
// RUN: enzymexlamlir-opt --lower-enzymexla-math %s | stablehlo-translate --interpret

// enzyme.binomial_progress returns the Revolve *advance distance*: with
// beta(s,t) = C(s+t,t) and t minimal such that beta(budget,t) >= num_steps, the
// midpoint of [num_steps - beta(budget-1,t), beta(budget,t-1)] clamped to
// [1, num_steps-1], then capped at num_steps - (budget-1). It grows like
// num_steps, not like num_steps^(1/budget).
//
// This is the tensor form of the op; Enzyme's own lower-enzyme-binomial-progress
// lowers the scalar form onto scf/arith and must produce the same values. See
// enzyme/test/MLIR/Passes/lower_binomial_progress.mlir.

// Constant operands fold to a plain constant. For (9, 3): t = 2, since
// beta(3,1) = 4 < 9 <= beta(3,2) = 10; the window is
// [9 - beta(2,2), beta(3,1)] = [9 - 6, 4] = [3, 4], midpoint 3.
func.func @cst() -> tensor<i64> {
  %n = stablehlo.constant dense<9> : tensor<i64>
  %s = stablehlo.constant dense<3> : tensor<i64>
  %r = enzyme.binomial_progress %n, %s : tensor<i64>
  return %r : tensor<i64>
}

// FOLD-LABEL: func.func @cst() -> tensor<i64> {
// FOLD-NEXT:    %[[R:.+]] = stablehlo.constant dense<3> : tensor<i64>
// FOLD-NEXT:    return %[[R]] : tensor<i64>
// FOLD-NOT:     enzyme.binomial_progress

// A budget of 1 advances the whole remaining stretch: with one checkpoint left
// it is replayed from there. This is also what makes the per-slot advances sum
// to exactly the trip count across `budget` slots, which is what lets a driver
// that iterates once per checkpoint still reach the end of the primal.
func.func @budget_one() -> tensor<i64> {
  %n = stablehlo.constant dense<40> : tensor<i64>
  %s = stablehlo.constant dense<1> : tensor<i64>
  %r = enzyme.binomial_progress %n, %s : tensor<i64>
  return %r : tensor<i64>
}

// FOLD-LABEL: func.func @budget_one() -> tensor<i64> {
// FOLD-NEXT:    %[[R:.+]] = stablehlo.constant dense<40> : tensor<i64>
// FOLD-NEXT:    return %[[R]] : tensor<i64>

// A single remaining step advances by one, never zero: a zero advance would
// leave the caller's loop without progress.
func.func @one_step() -> tensor<i64> {
  %n = stablehlo.constant dense<1> : tensor<i64>
  %s = stablehlo.constant dense<4> : tensor<i64>
  %r = enzyme.binomial_progress %n, %s : tensor<i64>
  return %r : tensor<i64>
}

// FOLD-LABEL: func.func @one_step() -> tensor<i64> {
// FOLD-NEXT:    %[[R:.+]] = stablehlo.constant dense<1> : tensor<i64>
// FOLD-NEXT:    return %[[R]] : tensor<i64>

// The advance is a constant fraction of the interval, not an inverse-binomial
// index: (400, 4) is in the hundreds. It was 11 when this op returned the
// repetition count, which left a 362-step stretch with no interior checkpoint
// and made the reverse pass quadratic.
func.func @large() -> tensor<i64> {
  %n = stablehlo.constant dense<400> : tensor<i64>
  %s = stablehlo.constant dense<4> : tensor<i64>
  %r = enzyme.binomial_progress %n, %s : tensor<i64>
  return %r : tensor<i64>
}

// FOLD-LABEL: func.func @large() -> tensor<i64> {
// FOLD-NEXT:    %[[R:.+]] = stablehlo.constant dense<282> : tensor<i64>
// FOLD-NEXT:    return %[[R]] : tensor<i64>

// Dynamic operands lower to the Revolve computation on stablehlo. The guard is
// a branch rather than a select because with budget <= 1 the loop below would
// leave beta at 1 and never terminate.
func.func @dyn(%n: tensor<i64>, %s: tensor<i64>) -> tensor<i64> {
  %r = enzyme.binomial_progress %n, %s : tensor<i64>
  return %r : tensor<i64>
}

// CHECK-LABEL: func.func @dyn(
// CHECK-SAME:    %[[N:.+]]: tensor<i64>, %[[S:.+]]: tensor<i64>
// CHECK-DAG:     %[[C2:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-DAG:     %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:         %[[NSM:.+]] = stablehlo.compare LE, %[[N]], %[[C1]]
// CHECK-NEXT:    %[[SSM:.+]] = stablehlo.compare LE, %[[S]], %[[C1]]
// CHECK-NEXT:    %[[G:.+]] = stablehlo.or %[[NSM]], %[[SSM]]
// CHECK-NEXT:    %{{.+}} = "stablehlo.if"(%[[G]]) ({
// CHECK-NEXT:      stablehlo.return %[[N]] : tensor<i64>
// CHECK-NEXT:    }, {
// beta = C(s+t, t), stepped from C(s+t-1, t-1) by *(s+t)/t.
// CHECK-NEXT:      %[[W:.+]]:2 = stablehlo.while(%[[T:.+]] = %[[C0]], %[[B:.+]] = %[[C1]])
// CHECK:           cond {
// CHECK-NEXT:        stablehlo.compare LT, %[[B]], %[[N]]
// CHECK-NEXT:        stablehlo.return
// CHECK:           } do {
// CHECK-NEXT:        %[[TN:.+]] = stablehlo.add %[[T]], %[[C1]]
// CHECK-NEXT:        %[[SPT:.+]] = stablehlo.add %[[S]], %[[TN]]
// CHECK-NEXT:        %[[MUL:.+]] = stablehlo.multiply %[[B]], %[[SPT]]
// CHECK-NEXT:        %[[DIV:.+]] = stablehlo.divide %[[MUL]], %[[TN]]
// CHECK-NEXT:        stablehlo.return %[[TN]], %[[DIV]]
// Window edges n - beta(s-1,t) and beta(s,t-1), clamped, then the midpoint.
// CHECK:           %[[SUM:.+]] = stablehlo.add %[[S]], %[[W]]#0
// CHECK-NEXT:      %[[LN:.+]] = stablehlo.multiply %[[W]]#1, %[[S]]
// CHECK-NEXT:      %[[LD:.+]] = stablehlo.divide %[[LN]], %[[SUM]]
// CHECK-NEXT:      %[[LO:.+]] = stablehlo.subtract %[[N]], %[[LD]]
// CHECK-NEXT:      %[[HN:.+]] = stablehlo.multiply %[[W]]#1, %[[W]]#0
// CHECK-NEXT:      %[[HI:.+]] = stablehlo.divide %[[HN]], %[[SUM]]
// CHECK-NEXT:      %[[CLO:.+]] = stablehlo.maximum %[[LO]], %[[C1]]
// CHECK-NEXT:      %[[NM1:.+]] = stablehlo.subtract %[[N]], %[[C1]]
// CHECK-NEXT:      %[[CHI:.+]] = stablehlo.minimum %[[HI]], %[[NM1]]
// CHECK-NEXT:      %[[ADD:.+]] = stablehlo.add %[[CLO]], %[[CHI]]
// CHECK-NEXT:      %[[MID:.+]] = stablehlo.divide %[[ADD]], %[[C2]]
// Cap so one step is left for each of the s-1 slots still to be placed,
// otherwise the advances can exhaust the interval before the slots run out.
// CHECK-NEXT:      %[[SM1:.+]] = stablehlo.subtract %[[S]], %[[C1]]
// CHECK-NEXT:      %[[CAP:.+]] = stablehlo.subtract %[[N]], %[[SM1]]
// CHECK-NEXT:      %[[CAPPED:.+]] = stablehlo.minimum %[[MID]], %[[CAP]]
// CHECK-NEXT:      %[[RES:.+]] = stablehlo.maximum %[[CAPPED]], %[[C1]]
// CHECK-NEXT:      stablehlo.return %[[RES]] : tensor<i64>
// CHECK-NOT:     enzyme.binomial_progress

// The lowered dynamic form and the constant folder must agree. Each case below
// runs the lowering through the interpreter and checks it against the value the
// folder produces for the same operands.
func.func @main() {
  %c9 = stablehlo.constant dense<9> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %r0 = enzyme.binomial_progress %c9, %c3 : tensor<i64>
  check.expect_eq_const %r0, dense<3> : tensor<i64>

  %c40 = stablehlo.constant dense<40> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %r1 = enzyme.binomial_progress %c40, %c1 : tensor<i64>
  check.expect_eq_const %r1, dense<40> : tensor<i64>

  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %r2 = enzyme.binomial_progress %c1, %c4 : tensor<i64>
  check.expect_eq_const %r2, dense<1> : tensor<i64>

  %c400 = stablehlo.constant dense<400> : tensor<i64>
  %r3 = enzyme.binomial_progress %c400, %c4 : tensor<i64>
  check.expect_eq_const %r3, dense<282> : tensor<i64>

  // The advances for successive slots sum to exactly the trip count: 282 of 400
  // with 4 slots, then 83 of the remaining 118 with 3, 27 of 35 with 2, and the
  // final 8 with 1. That 282 + 83 + 27 + 8 == 400 is what lets a driver walking
  // one slot per iteration reach the end of the primal.
  %c118 = stablehlo.constant dense<118> : tensor<i64>
  %r4 = enzyme.binomial_progress %c118, %c3 : tensor<i64>
  check.expect_eq_const %r4, dense<83> : tensor<i64>

  %c35 = stablehlo.constant dense<35> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %r5 = enzyme.binomial_progress %c35, %c2 : tensor<i64>
  check.expect_eq_const %r5, dense<27> : tensor<i64>

  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %r6 = enzyme.binomial_progress %c8, %c1 : tensor<i64>
  check.expect_eq_const %r6, dense<8> : tensor<i64>

  return
}
