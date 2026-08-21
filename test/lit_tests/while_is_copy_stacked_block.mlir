// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_is_copy_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Must not fire: block width 2 == step 2 (multi-element window per iteration).
module {
  func.func @stacked_block(%constraint: tensor<1x7xf64>, %trace_init: tensor<1x7xf64>) -> tensor<1x7xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %0:2 = stablehlo.while(%iter = %c0, %trace = %trace_init) : tensor<i64>, tensor<1x7xf64>
    cond {
      %1 = stablehlo.compare LT, %iter, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iter, %c1 : tensor<i64>
      %rc = stablehlo.multiply %iter, %c2 : tensor<i64>
      %block = stablehlo.dynamic_slice %constraint, %c0, %rc, sizes = [1, 2] : (tensor<1x7xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
      %wc = stablehlo.add %rc, %c1 : tensor<i64>
      %nt = stablehlo.dynamic_update_slice %trace, %block, %c0, %wc : (tensor<1x7xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x7xf64>
      stablehlo.return %1, %nt : tensor<i64>, tensor<1x7xf64>
    }
    return %0#1 : tensor<1x7xf64>
  }
}

// CHECK-LABEL: func.func @stacked_block(
// CHECK-NOT: stablehlo.scatter
// CHECK: %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[C2:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %[[C3:.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %[[W:.+]]:2 = stablehlo.while(%[[IV:.+]] = %[[C0]], %[[TR:.+]] = %arg1) : tensor<i64>, tensor<1x7xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT:   %[[CMP:.+]] = stablehlo.compare  LT, %[[IV]], %[[C3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:   stablehlo.return %[[CMP]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT:   %[[NEXT:.+]] = stablehlo.add %[[IV]], %[[C1]] : tensor<i64>
// CHECK-NEXT:   %[[RC:.+]] = stablehlo.multiply %[[IV]], %[[C2]] : tensor<i64>
// CHECK-NEXT:   %[[BLK:.+]] = stablehlo.dynamic_slice %arg0, %[[C0]], %[[RC]], sizes = [1, 2] : (tensor<1x7xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT:   %[[WC:.+]] = stablehlo.add %[[RC]], %[[C1]] : tensor<i64>
// CHECK-NEXT:   %[[NT:.+]] = stablehlo.dynamic_update_slice %[[TR]], %[[BLK]], %[[C0]], %[[WC]] : (tensor<1x7xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x7xf64>
// CHECK-NEXT:   stablehlo.return %[[NEXT]], %[[NT]] : tensor<i64>, tensor<1x7xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[W]]#1 : tensor<1x7xf64>

module {
  func.func @scalar_copy(%constraint: tensor<1x7xf64>, %trace_init: tensor<1x7xf64>) -> tensor<1x7xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %0:2 = stablehlo.while(%iter = %c0, %trace = %trace_init) : tensor<i64>, tensor<1x7xf64>
    cond {
      %1 = stablehlo.compare LT, %iter, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iter, %c1 : tensor<i64>
      %block = stablehlo.dynamic_slice %constraint, %c0, %iter, sizes = [1, 1] : (tensor<1x7xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %nt = stablehlo.dynamic_update_slice %trace, %block, %c0, %iter : (tensor<1x7xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x7xf64>
      stablehlo.return %1, %nt : tensor<i64>, tensor<1x7xf64>
    }
    return %0#1 : tensor<1x7xf64>
  }
}

// CHECK-LABEL: func.func @scalar_copy(
// CHECK-NOT: stablehlo.while
// CHECK: %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[SL:.+]] = stablehlo.slice %arg0 [0:1, 0:3] : (tensor<1x7xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[T0:.+]] = stablehlo.transpose %[[SL]], dims = [1, 0] : (tensor<1x3xf64>) -> tensor<3x1xf64>
// CHECK-NEXT: %[[T1:.+]] = stablehlo.transpose %[[T0]], dims = [1, 0] : (tensor<3x1xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[OFF:.+]] = stablehlo.add %[[C0]], %[[C0]] : tensor<i64>
// CHECK-NEXT: %[[R:.+]] = stablehlo.dynamic_update_slice %arg1, %[[T1]], %[[C0]], %[[OFF]] : (tensor<1x7xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x7xf64>
// CHECK-NEXT: return %[[R]] : tensor<1x7xf64>
