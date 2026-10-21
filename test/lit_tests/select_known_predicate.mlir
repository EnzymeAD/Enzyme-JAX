// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns="select_select_common_branch;select_known_predicate_operand;add_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: stablehlo-translate --interpret --allow-unregistered-dialect %s

func.func @select_common_true(%p: tensor<4xi1>, %q: tensor<4xi1>, %a: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %inner = stablehlo.select %q, %a, %c : tensor<4xi1>, tensor<4xf32>
  %r = stablehlo.select %p, %a, %inner : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_common_true
// CHECK-NEXT: %[[OR:.+]] = stablehlo.or %arg0, %arg1 : tensor<4xi1>
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %[[OR]], %arg2, %arg3
// CHECK-NEXT: return %[[S]]

func.func @select_common_false(%p: tensor<4xi1>, %q: tensor<4xi1>, %a: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %inner = stablehlo.select %q, %c, %a : tensor<4xi1>, tensor<4xf32>
  %r = stablehlo.select %p, %inner, %a : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_common_false
// CHECK-NEXT: %[[AND:.+]] = stablehlo.and %arg0, %arg1 : tensor<4xi1>
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %[[AND]], %arg3, %arg2
// CHECK-NEXT: return %[[S]]

// Different branches shared: nothing to merge.
func.func @select_no_common(%p: tensor<4xi1>, %q: tensor<4xi1>, %a: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %inner = stablehlo.select %q, %c, %a : tensor<4xi1>, tensor<4xf32>
  %r = stablehlo.select %p, %a, %inner : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_no_common
// CHECK-NEXT: %[[I:.+]] = stablehlo.select %arg1, %arg3, %arg2
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %arg0, %arg2, %[[I]]
// CHECK-NEXT: return %[[S]]

// Under p, select(p, 0, x) is 0, so (select(p, 0, x) + y) is y.
func.func @select_known_predicate(%p: tensor<4xi1>, %x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
  %z = stablehlo.select %p, %zero, %x : tensor<4xi1>, tensor<4xf32>
  %s = stablehlo.add %z, %y : tensor<4xf32>
  %r = stablehlo.select %p, %s, %x : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}
// CHECK-LABEL: func.func @select_known_predicate
// CHECK-NEXT: %[[S:.+]] = stablehlo.select %arg0, %arg2, %arg1
// CHECK-NEXT: return %[[S]]

// The false branch of the inner select is what the outer false branch sees;
// the intermediate add has another user and is cloned rather than changed.
func.func @select_known_predicate_false(%p: tensor<4xi1>, %x: tensor<4xf32>, %y: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %z = stablehlo.select %p, %x, %y : tensor<4xi1>, tensor<4xf32>
  %s = stablehlo.add %z, %y : tensor<4xf32>
  %r = stablehlo.select %p, %x, %s : tensor<4xi1>, tensor<4xf32>
  return %r, %s : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: func.func @select_known_predicate_false
// CHECK-DAG: %[[Z:.+]] = stablehlo.select %arg0, %arg1, %arg2
// CHECK-DAG: %[[S:.+]] = stablehlo.add %[[Z]], %arg2
// CHECK-DAG: %[[YY:.+]] = stablehlo.add %arg2, %arg2
// CHECK: %[[R:.+]] = stablehlo.select %arg0, %arg1, %[[YY]]
// CHECK-NEXT: return %[[R]], %[[S]]

func.func @main() {
  %p = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %q = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %c = stablehlo.constant dense<[-1.0, -2.0, -3.0, -4.0]> : tensor<4xf32>
  %r1 = func.call @select_common_true(%p, %q, %a, %c) : (tensor<4xi1>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e1 = stablehlo.constant dense<[1.0, 2.0, 3.0, -4.0]> : tensor<4xf32>
  check.expect_eq %r1, %e1 : tensor<4xf32>
  %r2 = func.call @select_common_false(%p, %q, %a, %c) : (tensor<4xi1>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e2 = stablehlo.constant dense<[-1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  check.expect_eq %r2, %e2 : tensor<4xf32>
  %r2b = func.call @select_no_common(%p, %q, %a, %c) : (tensor<4xi1>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e2b = stablehlo.constant dense<[1.0, -2.0, 3.0, 4.0]> : tensor<4xf32>
  check.expect_eq %r2b, %e2b : tensor<4xf32>
  %r3 = func.call @select_known_predicate(%p, %a, %c) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %e3 = stablehlo.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  check.expect_eq %r3, %e3 : tensor<4xf32>
  %r4:2 = func.call @select_known_predicate_false(%p, %a, %c) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %e4 = stablehlo.constant dense<[1.0, -4.0, 3.0, -8.0]> : tensor<4xf32>
  check.expect_eq %r4#0, %e4 : tensor<4xf32>
  %e4b = stablehlo.constant dense<[0.0, -4.0, 0.0, -8.0]> : tensor<4xf32>
  check.expect_eq %r4#1, %e4b : tensor<4xf32>
  return
}
