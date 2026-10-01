// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Reduce of a splat-true constant with `stablehlo.and` should fold to `true`.
// Regression test for https://github.com/EnzymeAD/Enzyme-JAX/issues/1084.
module {
  func.func @reduce_and_splat_true() -> tensor<i1> {
    %c_0 = stablehlo.constant dense<true> : tensor<1x140xi1>
    %c_7 = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.reduce(%c_0 init: %c_7) applies stablehlo.and across dimensions = [0, 1] : (tensor<1x140xi1>, tensor<i1>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// CHECK-LABEL: func.func @reduce_and_splat_true() -> tensor<i1> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    return %[[C]] : tensor<i1>
// CHECK-NEXT:  }

// Reduce of a splat-false constant with `stablehlo.or` should fold to `false`.
module {
  func.func @reduce_or_splat_false() -> tensor<i1> {
    %c_0 = stablehlo.constant dense<false> : tensor<1x140xi1>
    %c_7 = stablehlo.constant dense<false> : tensor<i1>
    %0 = stablehlo.reduce(%c_0 init: %c_7) applies stablehlo.or across dimensions = [0, 1] : (tensor<1x140xi1>, tensor<i1>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// CHECK-LABEL: func.func @reduce_or_splat_false() -> tensor<i1> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    return %[[C]] : tensor<i1>
// CHECK-NEXT:  }

// Negative: reduce with a non-constant input must not be folded.
module {
  func.func @reduce_and_nonconst(%arg0: tensor<1x140xi1>) -> tensor<i1> {
    %c_7 = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.reduce(%arg0 init: %c_7) applies stablehlo.and across dimensions = [0, 1] : (tensor<1x140xi1>, tensor<i1>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

// CHECK-LABEL: func.func @reduce_and_nonconst(
// CHECK:    stablehlo.reduce
