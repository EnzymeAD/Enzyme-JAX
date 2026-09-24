// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A loop that reads the very row of a carried buffer it writes back -- gemm's
// `C[i, :] = C[i, :] * beta + ...`. The read cannot be hoisted, the buffer
// being different every iteration, but read and write land on the same row, so
// the iterations touch disjoint rows and the loop is a map over the batched
// dimension. The chain feeding the write must therefore be batched.

func.func @rmw_rowwise(%a: tensor<16x8xf64>, %c: tensor<16x8xf64>, %beta: tensor<8xf64>) -> tensor<16x8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c0, %buf = %c) : tensor<i64>, tensor<16x8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %a, %iterArg, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x8xf64>) -> tensor<8xf64>
    %4 = stablehlo.multiply %3, %beta : tensor<8xf64>
    %5 = stablehlo.dynamic_slice %buf, %iterArg, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %6 = stablehlo.reshape %5 : (tensor<1x8xf64>) -> tensor<8xf64>
    %7 = stablehlo.add %4, %6 : tensor<8xf64>
    %8 = stablehlo.reshape %7 : (tensor<8xf64>) -> tensor<1x8xf64>
    %9 = stablehlo.dynamic_update_slice %buf, %8, %iterArg, %c0 : (tensor<16x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<16x8xf64>
    stablehlo.return %1, %9 : tensor<i64>, tensor<16x8xf64>
  }
  return %0#1 : tensor<16x8xf64>
}

// CHECK-LABEL: func.func @rmw_rowwise
// CHECK:         %[[BETA:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<8xf64>) -> tensor<16x8xf64>
// CHECK:         %[[SCALED:.+]] = stablehlo.multiply %arg0, %[[BETA]] : tensor<16x8xf64>
// CHECK:         stablehlo.while
// CHECK:         } do {
// CHECK:           stablehlo.dynamic_slice %[[SCALED]]
// CHECK-NOT:       stablehlo.multiply

// -----

// The same loop, but reading the row ahead of the one it writes: a genuine
// recurrence, whose iterations are not independent and which no amount of
// batching collapses. Nothing may be hoisted here.

func.func @rmw_shifted(%a: tensor<16x8xf64>, %c: tensor<16x8xf64>, %beta: tensor<8xf64>) -> tensor<16x8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c0, %buf = %c) : tensor<i64>, tensor<16x8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %a, %iterArg, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x8xf64>) -> tensor<8xf64>
    %4 = stablehlo.multiply %3, %beta : tensor<8xf64>
    %5 = stablehlo.dynamic_slice %buf, %1, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %6 = stablehlo.reshape %5 : (tensor<1x8xf64>) -> tensor<8xf64>
    %7 = stablehlo.add %4, %6 : tensor<8xf64>
    %8 = stablehlo.reshape %7 : (tensor<8xf64>) -> tensor<1x8xf64>
    %9 = stablehlo.dynamic_update_slice %buf, %8, %iterArg, %c0 : (tensor<16x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<16x8xf64>
    stablehlo.return %1, %9 : tensor<i64>, tensor<16x8xf64>
  }
  return %0#1 : tensor<16x8xf64>
}

// CHECK-LABEL: func.func @rmw_shifted
// CHECK:         stablehlo.while
// CHECK:         } do {
// CHECK:           %[[ROW:.+]] = stablehlo.dynamic_slice %arg0
// CHECK:           %[[FLAT:.+]] = stablehlo.reshape %[[ROW]]
// CHECK:           stablehlo.multiply %[[FLAT]], %arg2 : tensor<8xf64>

// -----

// `alpha * a[k]` is consumed inside the nested loop, so it is recomputed on
// every one of its iterations. The inner loop's own fission lifts it into the
// outer body, which is the loop-invariant code motion that matters. The outer
// loop then leaves it alone: its result is consumed inside the surviving inner
// loop, reaching neither the terminator nor a store to a carried buffer, so
// batching it across the outer trip count would only materialize a temporary
// without shrinking either loop. Both loops are scatters into a carried
// buffer, and both survive.

func.func @nested_consumer(%a: tensor<4xf64>, %b: tensor<8xf64>, %alpha: tensor<f64>, %out: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c0, %buf = %out) : tensor<i64>, tensor<8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %a, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4:2 = stablehlo.while(%j = %c0, %inner = %buf) : tensor<i64>, tensor<8xf64>
    cond {
      %5 = stablehlo.compare LT, %j, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %5 = stablehlo.add %j, %c1 : tensor<i64>
      %6 = stablehlo.multiply %alpha, %3 : tensor<f64>
      %7 = stablehlo.dynamic_slice %b, %j, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
      %8 = stablehlo.reshape %7 : (tensor<1xf64>) -> tensor<f64>
      %9 = stablehlo.multiply %6, %8 : tensor<f64>
      %10 = stablehlo.dynamic_slice %inner, %j, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
      %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
      %12 = stablehlo.add %11, %9 : tensor<f64>
      %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
      %14 = stablehlo.dynamic_update_slice %inner, %13, %j : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
      stablehlo.return %5, %14 : tensor<i64>, tensor<8xf64>
    }
    stablehlo.return %1, %4#1 : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK-LABEL: func.func @nested_consumer
// CHECK:         stablehlo.while
// The product is hoisted out of the inner loop, into the outer body, and stays
// a per-outer-iteration 8-element computation rather than a 4x8 one.
// CHECK:           stablehlo.multiply %{{.+}}, %{{.+}} : tensor<f64>
// CHECK:           %[[PROD:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<8xf64>
// CHECK:           stablehlo.while
// CHECK:             stablehlo.dynamic_slice %[[PROD]]
// CHECK-NOT:         stablehlo.multiply

// -----

// The computation forks and never merges back: `a[i] * beta + gamma` feeds two
// separate stores. A forked frontier is not hoisted as such -- growth follows
// it to the end and then falls back to the last point where it was a single
// value. So the prefix the two branches share is batched out and only the fork
// is left behind in the loop.

func.func @fork(%a: tensor<16x8xf64>, %beta: tensor<8xf64>, %gamma: tensor<8xf64>,
                %o1: tensor<16x8xf64>, %o2: tensor<16x8xf64>) -> (tensor<16x8xf64>, tensor<16x8xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %0:3 = stablehlo.while(%iterArg = %c0, %b1 = %o1, %b2 = %o2) : tensor<i64>, tensor<16x8xf64>, tensor<16x8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %a, %iterArg, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x8xf64>) -> tensor<8xf64>
    %4 = stablehlo.multiply %3, %beta : tensor<8xf64>
    %5 = stablehlo.add %4, %gamma : tensor<8xf64>
    %6 = stablehlo.multiply %5, %beta : tensor<8xf64>
    %7 = stablehlo.subtract %5, %gamma : tensor<8xf64>
    %8 = stablehlo.reshape %6 : (tensor<8xf64>) -> tensor<1x8xf64>
    %9 = stablehlo.reshape %7 : (tensor<8xf64>) -> tensor<1x8xf64>
    %10 = stablehlo.dynamic_update_slice %b1, %8, %iterArg, %c0 : (tensor<16x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<16x8xf64>
    %11 = stablehlo.dynamic_update_slice %b2, %9, %iterArg, %c0 : (tensor<16x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<16x8xf64>
    stablehlo.return %1, %10, %11 : tensor<i64>, tensor<16x8xf64>, tensor<16x8xf64>
  }
  return %0#1, %0#2 : tensor<16x8xf64>, tensor<16x8xf64>
}

// CHECK-LABEL: func.func @fork
// The shared prefix is hoisted, across all 16 rows at once.
// CHECK:         %[[SCALED:.+]] = stablehlo.multiply %arg0, %{{.+}} : tensor<16x8xf64>
// CHECK:         %[[SHIFTED:.+]] = stablehlo.add %[[SCALED]], %{{.+}} : tensor<16x8xf64>
// CHECK:         stablehlo.while
// CHECK:         } do {
// CHECK:           %[[ROW:.+]] = stablehlo.dynamic_slice %[[SHIFTED]]
// The two branches of the fork stay behind.
// CHECK:           %[[FLAT:.+]] = stablehlo.reshape %[[ROW]]
// CHECK:           stablehlo.multiply %[[FLAT]], %arg1 : tensor<8xf64>
// CHECK:           stablehlo.subtract %[[FLAT]], %arg2 : tensor<8xf64>

// -----

// The same fork, but neither branch ends anywhere worth hoisting to -- both add
// the row AHEAD of the one they write, a recurrence in each. The shared prefix
// is still batched out: what makes it pay is that the fork consumes it twice,
// which is independent of what the branches go on to do. Contrast @rmw_shifted,
// where growth is blocked outright and there is no second consumer to amortize
// against, so nothing moves.

func.func @fork_recurrent(%a: tensor<16x8xf64>, %beta: tensor<8xf64>, %gamma: tensor<8xf64>,
                          %o1: tensor<16x8xf64>, %o2: tensor<16x8xf64>) -> (tensor<16x8xf64>, tensor<16x8xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %0:3 = stablehlo.while(%iterArg = %c0, %b1 = %o1, %b2 = %o2) : tensor<i64>, tensor<16x8xf64>, tensor<16x8xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c1 : tensor<i64>
    %2 = stablehlo.dynamic_slice %a, %iterArg, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x8xf64>) -> tensor<8xf64>
    %4 = stablehlo.multiply %3, %beta : tensor<8xf64>
    %5 = stablehlo.add %4, %gamma : tensor<8xf64>
    %6 = stablehlo.multiply %5, %beta : tensor<8xf64>
    %7 = stablehlo.subtract %5, %gamma : tensor<8xf64>
    %8 = stablehlo.dynamic_slice %b1, %1, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %9 = stablehlo.reshape %8 : (tensor<1x8xf64>) -> tensor<8xf64>
    %10 = stablehlo.add %6, %9 : tensor<8xf64>
    %11 = stablehlo.dynamic_slice %b2, %1, %c0, sizes = [1, 8] : (tensor<16x8xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
    %12 = stablehlo.reshape %11 : (tensor<1x8xf64>) -> tensor<8xf64>
    %13 = stablehlo.add %7, %12 : tensor<8xf64>
    %14 = stablehlo.reshape %10 : (tensor<8xf64>) -> tensor<1x8xf64>
    %15 = stablehlo.reshape %13 : (tensor<8xf64>) -> tensor<1x8xf64>
    %16 = stablehlo.dynamic_update_slice %b1, %14, %iterArg, %c0 : (tensor<16x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<16x8xf64>
    %17 = stablehlo.dynamic_update_slice %b2, %15, %iterArg, %c0 : (tensor<16x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<16x8xf64>
    stablehlo.return %1, %16, %17 : tensor<i64>, tensor<16x8xf64>, tensor<16x8xf64>
  }
  return %0#1, %0#2 : tensor<16x8xf64>, tensor<16x8xf64>
}

// CHECK-LABEL: func.func @fork_recurrent
// CHECK:         %[[SCALED:.+]] = stablehlo.multiply %arg0, %{{.+}} : tensor<16x8xf64>
// CHECK:         %[[SHIFTED:.+]] = stablehlo.add %[[SCALED]], %{{.+}} : tensor<16x8xf64>
// CHECK:         stablehlo.while
// CHECK:         } do {
// CHECK:           %[[ROW:.+]] = stablehlo.dynamic_slice %[[SHIFTED]]
// CHECK:           %[[FLAT:.+]] = stablehlo.reshape %[[ROW]]
// CHECK:           stablehlo.multiply %[[FLAT]], %arg1 : tensor<8xf64>
// CHECK:           stablehlo.subtract %[[FLAT]], %arg2 : tensor<8xf64>
