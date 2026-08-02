// RUN: enzymexlamlir-opt --canonicalize-elementwise-shapes --split-input-file %s | FileCheck %s

// Reduced from grad(KAN with use_base_act=true) in Reactant.jl's benchmark/kan.
// The softsign chain is written in tensor<400000x1> because reshape propagation
// hoisted the flatten in the primal, but after AD the reverse pass reads two of
// its intermediates at tensor<10000x40>. Rewriting the chain to tensor<10000x40>
// takes the boundary from three reshapes down to one.
module {
  func.func @softsign_split_shape(%pre : tensor<40x10000xf32>, %grid : tensor<10xf32>, %ct : tensor<10000x40xf32>) -> (tensor<400000x10xf32>, tensor<10000x40xf32>) {
    %one = stablehlo.constant dense<1.000000e+00> : tensor<400000x1xf32>
    %1 = stablehlo.transpose %pre, dims = [1, 0] : (tensor<40x10000xf32>) -> tensor<10000x40xf32>
    %2 = stablehlo.reshape %1 : (tensor<10000x40xf32>) -> tensor<400000x1xf32>
    %3 = stablehlo.abs %2 : tensor<400000x1xf32>
    %4 = stablehlo.add %one, %3 : tensor<400000x1xf32>
    %6 = stablehlo.divide %2, %4 : tensor<400000x1xf32>
    %8 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<400000x1xf32>) -> tensor<400000x10xf32>
    %9 = stablehlo.broadcast_in_dim %grid, dims = [1] : (tensor<10xf32>) -> tensor<400000x10xf32>
    %fwd = stablehlo.subtract %8, %9 : tensor<400000x10xf32>
    %5 = stablehlo.reshape %4 : (tensor<400000x1xf32>) -> tensor<10000x40xf32>
    %7 = stablehlo.reshape %6 : (tensor<400000x1xf32>) -> tensor<10000x40xf32>
    %10 = stablehlo.divide %ct, %5 : tensor<10000x40xf32>
    %11 = stablehlo.multiply %10, %7 : tensor<10000x40xf32>
    return %fwd, %11 : tensor<400000x10xf32>, tensor<10000x40xf32>
  }
}

// CHECK-LABEL:   func.func @softsign_split_shape(
// CHECK-NEXT:      %[[T:.*]] = stablehlo.transpose %{{.*}}, dims = [1, 0] : (tensor<40x10000xf32>) -> tensor<10000x40xf32>
// CHECK-NEXT:      %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<10000x40xf32>
// The whole chain now lives in the shape the reverse pass wants.
// CHECK-NEXT:      %[[ABS:.*]] = stablehlo.abs %[[T]] : tensor<10000x40xf32>
// CHECK-NEXT:      %[[ADD:.*]] = stablehlo.add %[[CST]], %[[ABS]] : tensor<10000x40xf32>
// CHECK-NEXT:      %[[DIV:.*]] = stablehlo.divide %[[T]], %[[ADD]] : tensor<10000x40xf32>
// A single boundary reshape remains, for the one forward consumer.
// CHECK-NEXT:      %[[RS:.*]] = stablehlo.reshape %[[DIV]] : (tensor<10000x40xf32>) -> tensor<400000x1xf32>
// CHECK-NEXT:      %[[BC:.*]] = stablehlo.broadcast_in_dim %[[RS]]
// CHECK-NEXT:      %[[BG:.*]] = stablehlo.broadcast_in_dim %{{.*}}
// CHECK-NEXT:      %[[FWD:.*]] = stablehlo.subtract %[[BC]], %[[BG]]
// The reverse consumers read the chain directly, with no reshape at all.
// CHECK-NEXT:      %[[R0:.*]] = stablehlo.divide %{{.*}}, %[[ADD]] : tensor<10000x40xf32>
// CHECK-NEXT:      %[[R1:.*]] = stablehlo.multiply %[[R0]], %[[DIV]] : tensor<10000x40xf32>
// CHECK-NEXT:      return %[[FWD]], %[[R1]]

// -----

// A chain that is already uniform, with a single reshaped consumer, must be
// left alone: rewriting it would only move the reshape, not remove it.
module {
  func.func @already_uniform(%a : tensor<10000x40xf32>) -> tensor<400000x1xf32> {
    %0 = stablehlo.abs %a : tensor<10000x40xf32>
    %1 = stablehlo.negate %0 : tensor<10000x40xf32>
    %2 = stablehlo.reshape %1 : (tensor<10000x40xf32>) -> tensor<400000x1xf32>
    return %2 : tensor<400000x1xf32>
  }
}

// CHECK-LABEL:   func.func @already_uniform(
// CHECK-NEXT:      %[[A:.*]] = stablehlo.abs %{{.*}} : tensor<10000x40xf32>
// CHECK-NEXT:      %[[N:.*]] = stablehlo.negate %[[A]] : tensor<10000x40xf32>
// CHECK-NEXT:      %[[R:.*]] = stablehlo.reshape %[[N]] : (tensor<10000x40xf32>) -> tensor<400000x1xf32>
// CHECK-NEXT:      return %[[R]]

// -----

// Sinking the chain to the input shape removes the input reshape and lets the
// elementwise consumer read it directly; only the escaping value keeps one.
module {
  func.func @input_shape_wins(%a : tensor<40x10xf32>) -> (tensor<40x10xf32>, tensor<400x1xf32>) {
    %0 = stablehlo.reshape %a : (tensor<40x10xf32>) -> tensor<400x1xf32>
    %1 = stablehlo.abs %0 : tensor<400x1xf32>
    %2 = stablehlo.negate %1 : tensor<400x1xf32>
    %3 = stablehlo.reshape %2 : (tensor<400x1xf32>) -> tensor<40x10xf32>
    %4 = stablehlo.exponential %3 : tensor<40x10xf32>
    return %4, %2 : tensor<40x10xf32>, tensor<400x1xf32>
  }
}

// CHECK-LABEL:   func.func @input_shape_wins(
// CHECK-NEXT:      %[[A:.*]] = stablehlo.abs %{{.*}} : tensor<40x10xf32>
// CHECK-NEXT:      %[[N:.*]] = stablehlo.negate %[[A]] : tensor<40x10xf32>
// CHECK-NEXT:      %[[R:.*]] = stablehlo.reshape %[[N]] : (tensor<40x10xf32>) -> tensor<400x1xf32>
// CHECK-NEXT:      %[[E:.*]] = stablehlo.exponential %[[N]] : tensor<40x10xf32>
// CHECK-NEXT:      return %[[E]], %[[R]]

// -----

// One reshape either way: no shape strictly wins, so the pass must not churn
// the IR.
module {
  func.func @no_strict_win(%a : tensor<400x1xf32>) -> (tensor<400x1xf32>, tensor<40x10xf32>) {
    %0 = stablehlo.abs %a : tensor<400x1xf32>
    %1 = stablehlo.negate %0 : tensor<400x1xf32>
    %2 = stablehlo.reshape %1 : (tensor<400x1xf32>) -> tensor<40x10xf32>
    %3 = stablehlo.exponential %2 : tensor<40x10xf32>
    return %1, %3 : tensor<400x1xf32>, tensor<40x10xf32>
  }
}

// CHECK-LABEL:   func.func @no_strict_win(
// CHECK-NEXT:      %[[A:.*]] = stablehlo.abs %{{.*}} : tensor<400x1xf32>
// CHECK-NEXT:      %[[N:.*]] = stablehlo.negate %[[A]] : tensor<400x1xf32>
// CHECK-NEXT:      %[[R:.*]] = stablehlo.reshape %[[N]] : (tensor<400x1xf32>) -> tensor<40x10xf32>
// CHECK-NEXT:      %[[E:.*]] = stablehlo.exponential %[[R]] : tensor<40x10xf32>
// CHECK-NEXT:      return %[[N]], %[[E]]
