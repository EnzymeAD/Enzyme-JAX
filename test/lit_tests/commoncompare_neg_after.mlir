// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// The negated twin appears after the matched compare: the rewrite replaces
// it with a negation of the earlier compare, which must be inserted after
// the compare it reads, not at the match point before it.
func.func @negpair(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi1>, tensor<4xi1>) {
  %le = stablehlo.compare LE, %a, %b, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %gt = stablehlo.compare GT, %a, %b, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %le, %gt : tensor<4xi1>, tensor<4xi1>
}

// CHECK-LABEL: func.func @negpair(
// CHECK-NEXT:    %[[LE:.+]] = stablehlo.compare LE, %arg0, %arg1, SIGNED
// CHECK-NEXT:    %[[NOT:.+]] = stablehlo.not %[[LE]]
// CHECK-NEXT:    return %[[LE]], %[[NOT]]
